import pytest

from model_gateway.managed_auth import tables

pytestmark = pytest.mark.unit


def test_tables_no_longer_declare_the_project_period_columns():
    assert not {"period", "base_limit_usd"} & set(tables.project_policies.c.keys())
    assert not {"period_start", "period_end"} & set(
        tables.project_limit_requests.c.keys()
    )
