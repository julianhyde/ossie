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

This version expands a query that reads one relation of a model, with
no joins: it replaces the relation with the dataset's source, and each
``AGG(m)`` (or its synonym ``MEASURE(m)``) with the expression of
metric ``m``. On one relation, no join repeats rows, so that is the
reference evaluation of spec Section 9. A view is replaced with its
query, as a subquery.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from ossie.models import OssieDataset, OssieDialect, OssieDocument, OssieExpression
from sqlglot import exp

# The dialects whose expressions the expander can read, in order of
# preference.
DIALECTS = (OssieDialect.OSSIE_SQL_2026, OssieDialect.ANSI_SQL)


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
        statements = sqlglot.parse(sql, read="duckdb")
    except sqlglot.errors.ParseError:
        return False
    return any(
        catalog.contains(table)
        for statement in statements
        if statement is not None
        for table in statement.find_all(exp.Table)
    )


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
    for select in list(tree.find_all(exp.Select)):
        _expand_select(select, catalog)
    for node in tree.walk():
        if _measure_ref(node) is not None:
            raise ValueError(f"cannot evaluate {node.sql(dialect='duckdb')}")
    return tree.sql(dialect="duckdb", pretty=True)


def _parse(sql: str) -> exp.Expression:
    try:
        return sqlglot.parse_one(sql, read="duckdb")
    except sqlglot.errors.ParseError as e:
        raise ValueError(str(e).splitlines()[0]) from None


def _expand_select(select: exp.Select, catalog: Catalog) -> None:
    """Expands a SELECT whose FROM clause reads a relation of a model."""
    from_ = select.args.get("from_")
    joins = select.args.get("joins") or []
    tables = [from_.this] if from_ else []
    tables += [join.this for join in joins]
    datasets = [
        (t, d)
        for t in tables
        if isinstance(t, exp.Table) and (d := catalog.dataset(t)) is not None
    ]
    if not datasets:
        return
    if len(tables) > 1:
        raise ValueError("view expander does not support joins yet")
    table, dataset = datasets[0]
    for field in dataset.fields or []:
        if _expression(field.expression).lower() != field.name.lower():
            raise ValueError(
                f"view expander does not support field expressions yet: {field.name}"
            )

    # Replace each AGG(m) in this SELECT, but not in its subqueries,
    # with the expression of metric m.
    alias = table.alias_or_name.lower()
    metrics = {m.name.lower(): m for m in dataset.metrics or []}
    for node in list(select.walk()):
        ref = _measure_ref(node)
        if ref is None or node.find_ancestor(exp.Select) is not select:
            continue
        metric = None
        if isinstance(ref, exp.Column) and ref.table.lower() in ("", alias):
            metric = metrics.get(ref.name.lower())
        if metric is None:
            raise ValueError(f"unknown measure {ref.sql(dialect='duckdb')}")
        node.replace(_parse(_expression(metric.expression)))

    # Replace the relation with the dataset's source. Keep the alias, so
    # that references such as "emp.deptno" are still valid.
    source = exp.to_table(dataset.source)
    alias_ = table.args.get("alias")
    if alias_ is None and source.name.lower() != dataset.name.lower():
        alias_ = exp.TableAlias(this=exp.to_identifier(dataset.name))
    if alias_ is not None:
        source.set("alias", alias_.copy())
    table.replace(source)


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
