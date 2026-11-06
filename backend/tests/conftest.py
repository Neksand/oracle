"""Test isolation: never touch the shipped database (assignment/oracle_dump.sql).

`app.db.database` reads ORACLE_DB_PATH at import time, so this must run before
any test module imports the app.
"""
from __future__ import annotations

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="oracle-tests-")
os.environ.setdefault("ORACLE_DB_PATH", os.path.join(_TMP, "oracle-test.db"))
