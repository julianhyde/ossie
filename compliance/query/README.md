<!--
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

Query compliance tests
======================

Test cases for the
[Relational Query Interface](../../core-spec/relational_semantics.md).

[`relational_semantics.ost`](relational_semantics.ost) connects to the
SCOTT schema and defines an Ossie model over it, `scott_model`, with
dataset-scoped metrics. Each test case has three parts: an Ossie query
that uses `AGG()`, its expansion into an equivalent DuckDB query over
the SCOTT tables, and the result of executing the expansion.

The file is an `.ost` script
([#501](https://github.com/apache/ossie/issues/501)). The `ost` runner
expands each Ossie query
([#502](https://github.com/apache/ossie/issues/502)), executes the
expansion in DuckDB, and checks that the expansion and the result are
as expected. To run it:

```sh
cd ost
uv run ost ../compliance/query
```
