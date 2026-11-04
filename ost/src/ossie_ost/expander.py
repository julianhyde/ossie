# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Expands a query over models and measure views into plain SQL.

A query that reads a relation of a model (see ``modelInline``) or a
view (see ``createView``) may use measures, which DuckDB cannot
execute. The view expander translates such a query into an
equivalent query over the underlying tables (apache/ossie#502).

This version expands a query whose SELECTs each read at most one
relation with measures, with no joins: it replaces a relation of a
model with the dataset's source, and each ``AGG(m)`` (or its synonym
``MEASURE(m)``) with the expression of metric ``m``. On one relation,
no join repeats rows, so that is the reference evaluation of spec
Section 9. A subquery or CTE that does not evaluate a measure passes
it through, projecting the columns that the measure's expression
needs. A view is replaced with its query, as a subquery.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from ossie.models import OssieDataset, OssieDialect, OssieDocument, OssieExpression
from sqlglot import exp
from sqlglot.dialects.duckdb import DuckDB

# The dialects whose expressions the expander can read, in order of
# preference.
DIALECTS = (OssieDialect.OSSIE_SQL_2026, OssieDialect.ANSI_SQL)


class Sql(DuckDB):
    """DuckDB SQL, as the expander reads and writes it.

    sqlglot writes ``NULLS FIRST`` or ``NULLS LAST`` only if it differs
    from DuckDB's default. This dialect writes it if the query did, so
    that the expansion does not rely on the default of the database.
    """

    class Parser(DuckDB.Parser):
        def _parse_ordered(self, parse_method=None):
            ordered = super()._parse_ordered(parse_method)
            prev = self._tokens[self._index - 2 : self._index]
            if (
                ordered is not None
                and len(prev) == 2
                and prev[0].text.upper() == "NULLS"
                and prev[1].text.upper() in ("FIRST", "LAST")
            ):
                ordered.meta["nulls"] = prev[1].text.upper()
            return ordered

    class Generator(DuckDB.Generator):
        def ordered_sql(self, expression: exp.Ordered) -> str:
            sql = super().ordered_sql(expression)
            nulls = expression.meta.get("nulls")
            if nulls and not sql.upper().endswith(" NULLS " + nulls):
                sql += " NULLS " + nulls
            return sql


@dataclass
class Catalog:
    """The models and views that a script has defined.

    ``models`` maps a model's name to the model. Each dataset of a
    model is a relation in a schema with the name of the model, such
    as ``scott_model.emp``. ``views`` maps a view's name to the text
    of its query.
    """

    models: dict[str, OssieDocument] = field(default_factory=dict)
    views: dict[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.models or self.views)

    def add_model(self, name: str, yaml: str) -> None:
        model = OssieDocument.from_ossie_yaml(yaml)
        if model.name != name:
            raise ValueError(f"model is named {model.name!r}, not {name!r}")
        self.models[name.lower()] = model

    def add_view(self, name: str, sql: str) -> None:
        self.views[name.lower()] = sql

    def contains(self, table: exp.Table) -> bool:
        """Returns whether a table reference is to a relation in this
        catalog."""
        return self.view(table) is not None or self.dataset(table) is not None

    def view(self, table: exp.Table) -> str | None:
        """Returns the query of the view that a table reference is to,
        or None."""
        if table.db or table.catalog:
            return None
        return self.views.get(table.name.lower())

    def dataset(self, table: exp.Table) -> OssieDataset | None:
        """Returns the dataset that a table reference is to, or None."""
        if not table.db or table.catalog:
            return None
        model = self.models.get(table.db.lower())
        if model is None:
            return None
        name = table.name.lower()
        return next((d for d in model.datasets if d.name.lower() == name), None)


def references(sql: str, catalog: Catalog) -> bool:
    """Returns whether ``sql`` reads a relation in ``catalog``, and
    therefore needs to be expanded.

    Returns False if ``sql`` cannot be parsed; DuckDB will report the
    error when it executes ``sql``.
    """
    if not catalog:
        return False
    try:
        statements = sqlglot.parse(sql, read=Sql)
    except sqlglot.errors.ParseError:
        return False
    return any(
        catalog.contains(table)
        for statement in statements
        if statement is not None
        for table in statement.find_all(exp.Table)
    )


# The measures of a relation: each measure's name (in lower case) and
# its expression, over the relation's columns; or, if the measure has no
# expression that the expander can read, the error to raise if a query
# evaluates it.
Measures = dict[str, exp.Expression | ValueError]


def expand(sql: str, catalog: Catalog) -> str:
    """Returns a query, without measures, equivalent to ``sql``.

    Raises ValueError if ``sql`` is invalid, or uses a feature that the
    expander does not support yet.
    """
    tree = _parse(sql)
    for table in list(tree.find_all(exp.Table)):
        view = catalog.view(table)
        if view is not None:
            alias = table.args.get("alias") or exp.TableAlias(this=table.this.copy())
            table.replace(exp.Subquery(this=_parse(view), alias=alias))
    ctes = {cte.alias_or_name.lower(): cte.this for cte in tree.find_all(exp.CTE)}
    # Expand each SELECT after the SELECTs within it, so that the
    # measures of its subqueries and CTEs are known.
    outputs: dict[int, Measures] = {}
    for select in reversed(list(tree.find_all(exp.Select))):
        outputs[id(select)] = _expand_select(select, catalog, ctes, outputs)
    for node in tree.walk():
        if _measure_ref(node) is not None:
            raise ValueError(f"cannot evaluate {node.sql(dialect=Sql)}")
    return tree.sql(dialect=Sql, pretty=True)


def _parse(sql: str) -> exp.Expression:
    try:
        return sqlglot.parse_one(sql, read=Sql)
    except sqlglot.errors.ParseError as e:
        raise ValueError(str(e).splitlines()[0]) from None


def _expand_select(
    select: exp.Select,
    catalog: Catalog,
    ctes: dict[str, exp.Expression],
    outputs: dict[int, Measures],
) -> Measures:
    """Expands a SELECT, and returns the measures of its output.

    If the SELECT evaluates measures, it replaces each ``AGG(m)`` with
    the expression of ``m``, and its output has no measures. Otherwise,
    the measures of its input that it projects become measures of its
    output; the SELECT projects, instead of each measure, the columns
    that the measure's expression needs.
    """
    from_ = select.args.get("from_")
    joins = select.args.get("joins") or []
    relations = [from_.this] if from_ else []
    relations += [join.this for join in joins]
    inputs = [(r, m) for r in relations if (m := _measures(r, catalog, ctes, outputs))]
    if not inputs:
        return {}
    if len(relations) > 1:
        raise ValueError("view expander does not support joins yet")
    relation, measures = inputs[0]
    alias = relation.alias_or_name.lower()
    _replace_dataset(relation, catalog)

    # Replace each AGG(m) in this SELECT, but not in its subqueries,
    # with the expression of measure m.
    evaluated = False
    for node in list(select.walk()):
        ref = _measure_ref(node)
        if ref is None or node.find_ancestor(exp.Select) is not select:
            continue
        expression = None
        if isinstance(ref, exp.Column) and ref.table.lower() in ("", alias):
            expression = measures.get(ref.name.lower())
        if expression is None:
            raise ValueError(f"unknown measure {ref.sql(dialect=Sql)}")
        if isinstance(expression, ValueError):
            raise expression
        node.replace(expression.copy())
        evaluated = True
    if evaluated:
        return {}
    return _project_measures(select, alias, measures)


def _measures(
    relation: exp.Expression,
    catalog: Catalog,
    ctes: dict[str, exp.Expression],
    outputs: dict[int, Measures],
) -> Measures:
    """Returns the measures of a relation in a FROM clause."""
    if isinstance(relation, exp.Subquery):
        return outputs.get(id(relation.this), {})
    if not isinstance(relation, exp.Table):
        return {}
    dataset = catalog.dataset(relation)
    if dataset is not None:
        for field in dataset.fields or []:
            if _expression(field.expression).lower() != field.name.lower():
                raise ValueError(
                    f"view expander does not support field expressions yet: {field.name}"
                )
        return {m.name.lower(): _metric(m.expression) for m in dataset.metrics or []}
    if not relation.db and (cte := ctes.get(relation.name.lower())) is not None:
        return outputs.get(id(cte), {})
    return {}


def _replace_dataset(relation: exp.Expression, catalog: Catalog) -> None:
    """If ``relation`` is a relation of a model, replaces it with the
    dataset's source. Keeps the alias, so that references such as
    ``emp.deptno`` are still valid."""
    if not isinstance(relation, exp.Table):
        return
    dataset = catalog.dataset(relation)
    if dataset is None:
        return
    source = exp.to_table(dataset.source)
    alias = relation.args.get("alias")
    if alias is None and source.name.lower() != dataset.name.lower():
        alias = exp.TableAlias(this=exp.to_identifier(dataset.name))
    if alias is not None:
        source.set("alias", alias.copy())
    relation.replace(source)


def _project_measures(select: exp.Select, alias: str, measures: Measures) -> Measures:
    """Returns the measures that an unaggregated SELECT projects, and
    replaces each in its SELECT list with the columns that it needs.

    ``*`` projects every measure, and every column, so the measures'
    expressions are unchanged.
    """
    output: Measures = {}
    items: list[exp.Expression] = []
    star = False
    for item in select.expressions:
        inner = item.unalias()
        if isinstance(inner, exp.Star) or (
            isinstance(inner, exp.Column) and isinstance(inner.this, exp.Star)
        ):
            star = True
            output.update(measures)
            items.append(item)
        elif (
            isinstance(inner, exp.Column)
            and inner.table.lower() in ("", alias)
            and inner.name.lower() in measures
        ):
            output[item.alias_or_name.lower()] = measures[inner.name.lower()]
        else:
            items.append(item)
    if star:
        select.set("expressions", items)
        return output

    # Project each column that a measure needs, under its own name if
    # the SELECT does not use that name for something else.
    projected = {item.alias_or_name.lower(): item for item in items}
    for name, expression in output.items():
        if isinstance(expression, ValueError):
            continue
        expression = expression.copy()
        for column in list(expression.find_all(exp.Column)):
            existing = projected.get(column.name.lower())
            if isinstance(existing, exp.Column) and existing.name.lower() == column.name.lower():
                continue
            new_name = column.name
            if existing is not None:
                new_name = f"{name}${column.name}"
            if new_name.lower() not in projected:
                item = exp.column(column.name)
                if new_name != column.name:
                    item = exp.alias_(item, new_name, quoted=True)
                items.append(item)
                projected[new_name.lower()] = item
            column.replace(exp.column(exp.to_identifier(new_name, quoted=new_name != column.name)))
        output[name] = expression
    select.set("expressions", items or [exp.Star()])
    return output


def _metric(expression: OssieExpression) -> exp.Expression | ValueError:
    """Parses the expression of a metric. Removes the qualifier of each
    column, so that the expression is valid whatever the relation's
    alias."""
    try:
        tree = _parse(_expression(expression))
    except ValueError as e:
        return e
    for column in tree.find_all(exp.Column):
        column.set("table", None)
    return tree


def _measure_ref(node: exp.Expression) -> exp.Expression | None:
    """If ``node`` is ``AGG(m)`` or ``MEASURE(m)``, returns ``m``."""
    if isinstance(node, exp.Agg):
        return node.this
    if (
        isinstance(node, exp.Anonymous)
        and node.name.upper() == "MEASURE"
        and len(node.expressions) == 1
    ):
        return node.expressions[0]
    return None


def _expression(expression: OssieExpression) -> str:
    """Returns the text of an expression in a dialect that the
    expander can read."""
    by_dialect = {d.dialect: d.expression for d in expression.dialects}
    for dialect in DIALECTS:
        if dialect in by_dialect:
            return by_dialect[dialect]
    raise ValueError(
        "no expression in dialect " + " or ".join(d.value for d in DIALECTS)
    )
