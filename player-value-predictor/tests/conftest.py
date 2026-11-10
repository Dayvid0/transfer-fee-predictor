import duckdb
import pytest

from fixture_db import make_fixture


@pytest.fixture
def con():
    """A fresh in-memory database for every test."""
    connection = duckdb.connect(":memory:")
    make_fixture(connection)
    yield connection
    connection.close()
