import pytest  # pyright: ignore[reportMissingImports]

import model_gateway.usage_ledger.redshift_schema as redshift_schema


def test_schema_statement_batches_keep_dimension_reseed_atomic() -> None:
    batches = redshift_schema.schema_statement_batches()
    seed = redshift_schema.analytics_dimensions_seed_statements()

    assert seed in batches
    assert all(len(batch) == 1 for batch in batches if batch != seed)
    assert batches[-1] == seed
    assert seed[0].startswith("delete from gateway_usage.analytics_dimensions")
    assert seed[1].startswith("insert into gateway_usage.analytics_dimensions")


def test_fast_aggregate_dimensions_are_derived_from_dimension_registry():
    fast_dimensions = tuple(
        dimension
        for dimension in redshift_schema.ANALYTICS_DIMENSIONS
        if dimension.is_in_fast_agg
    )

    assert redshift_schema.FAST_AGGREGATE_DIMENSIONS == tuple(
        dimension.dimension_kind for dimension in fast_dimensions
    )
    assert len(redshift_schema.FAST_AGGREGATE_DIMENSIONS) == len(
        set(redshift_schema.FAST_AGGREGATE_DIMENSIONS)
    )
    assert all(dimension.source_column for dimension in fast_dimensions)
    assert all(dimension.fact_column for dimension in fast_dimensions)
    assert all(dimension.aggregate_column_type for dimension in fast_dimensions)
    assert redshift_schema.is_reserved_dimension_value(
        redshift_schema.MISSING_DIMENSION_VALUE
    )
    assert not redshift_schema.is_reserved_dimension_value("actual-value")


def test_aggregate_tables_are_defined_per_grain_without_non_additive_metrics():
    assert redshift_schema.AGGREGATE_TABLE_BY_GRAIN == {
        "5m": "usage_agg_5m",
        "1h": "usage_agg_1h",
        "1d": "usage_agg_1d",
    }

    for grain in redshift_schema.AGGREGATE_GRAINS:
        ddl = redshift_schema.aggregate_table_ddl(grain)
        assert f"create table if not exists gateway_usage.usage_agg_{grain}" in ddl
        assert "bucket_start_utc timestamptz not null" in ddl
        for dimension in redshift_schema.ANALYTICS_DIMENSIONS:
            if dimension.is_in_fast_agg:
                assert (
                    f"{dimension.source_column} {dimension.aggregate_column_type} not null"
                    in ddl
                )
        assert "duration_seconds_sum decimal(38,6) not null" in ddl
        assert "duration_observed_count bigint not null" in ddl
        assert "cost_usd_sum decimal(38,12) not null" in ddl
        assert "percentile_cont" not in ddl.lower()
        assert "run_count" not in ddl.lower()
        assert "question_count" not in ddl.lower()


def test_fact_staging_debug_and_dimension_table_ddls_are_specific():
    staging_ddl = redshift_schema.usage_events_staging_table_ddl()
    facts_ddl = redshift_schema.usage_events_table_ddl()
    debug_ddl = redshift_schema.usage_event_debug_table_ddl()
    dimension_values_ddl = redshift_schema.usage_dimension_values_table_ddl()
    analytics_dimensions_ddl = redshift_schema.analytics_dimensions_table_ddl()

    assert (
        "create table if not exists gateway_usage.usage_events_staging" in staging_ddl
    )
    assert "batch_id varchar(128) not null" in staging_ddl
    assert "sortkey (batch_id, completed_at)" in staging_ddl
    staging_column_block = staging_ddl.split("(\n", 1)[1].split("\n)\n", 1)[0]
    staging_column_names = tuple(
        line.strip().split(maxsplit=1)[0]
        for line in staging_column_block.splitlines()
        if line.strip()
    )
    assert staging_column_names == redshift_schema.USAGE_EVENTS_STAGING_COLUMN_NAMES

    assert "create table if not exists gateway_usage.usage_events" in facts_ddl
    assert "batch_id varchar(128)" not in facts_ddl
    for column in (
        "usage_event_id varchar(128) not null",
        "completed_at timestamptz not null",
        "completed_date date not null",
        "completed_hour timestamp not null",
        "run_id varchar(65535)",
        "question_id varchar(65535)",
        "query_id varchar(65535)",
        "query_id_normalized varchar(65535)",
        "requested_model_key varchar(2048) not null",
        "provider varchar(256) not null",
        "provider_endpoint varchar(64) not null",
        "param_group varchar(512)",
        "config_hash varchar(256)",
        "benchmark_name varchar(1024)",
        "agent_name varchar(1024)",
        "identity_email varchar(1024)",
        "api_key_fingerprint varchar(512)",
        "input_tokens bigint not null default 0",
        "output_tokens bigint not null default 0",
        "reasoning_tokens bigint not null default 0",
        "cache_read_tokens bigint not null default 0",
        "cache_write_tokens bigint not null default 0",
        "total_input_tokens bigint not null default 0",
        "total_output_tokens bigint not null default 0",
        "duration_seconds decimal(18,6)",
        "cost_usd decimal(38,12) not null default 0",
        "finish_reason varchar(256)",
        "finish_reason_raw varchar(2048)",
        "schema_version integer not null",
        "metadata_schema_version integer not null",
        "normalization_version varchar(64) not null",
        "usage_shard varchar(8)",
        "source_pk varchar(2048)",
        "source_sk varchar(2048)",
        "loaded_at timestamptz not null default getdate()",
    ):
        assert column in facts_ddl

    assert "create table if not exists gateway_usage.usage_event_debug" in debug_ddl
    for column in (
        "usage_event_id varchar(128) not null",
        "identity_json super",
        "provider_request_id varchar(65535)",
        "provider_response_id varchar(65535)",
        "config_redacted_json super",
        "metadata_json super",
        "finish_reason_json super",
        "performance_json super",
        "config_redacted_json_truncated boolean",
        "metadata_json_truncated boolean",
        "finish_reason_json_truncated boolean",
        "performance_json_truncated boolean",
        "loaded_at timestamptz not null default getdate()",
    ):
        assert column in debug_ddl

    assert (
        "create table if not exists gateway_usage.usage_dimension_values"
        in dimension_values_ddl
    )
    for column in (
        "dimension_kind varchar(64) not null",
        "dimension_value varchar(2048) not null",
        "first_seen_at timestamptz",
        "last_seen_at timestamptz",
        "request_count_7d bigint",
        "request_count_30d bigint",
        "cost_usd_30d decimal(38,12)",
        "updated_at timestamptz not null",
    ):
        assert column in dimension_values_ddl
    assert "sortkey (dimension_kind, dimension_value)" in dimension_values_ddl

    assert (
        "create table if not exists gateway_usage.analytics_dimensions"
        in analytics_dimensions_ddl
    )
    for column in (
        "dimension_kind varchar(64) not null",
        "source_column varchar(128) not null",
        "is_filterable boolean not null",
        "is_groupable boolean not null",
        "is_in_fast_agg boolean not null",
        "display_name varchar(128) not null",
        "sort_order integer not null",
    ):
        assert column in analytics_dimensions_ddl


def test_schema_statement_batches_include_required_tables_without_control_state():
    statements = tuple(
        sql for batch in redshift_schema.schema_statement_batches() for sql in batch
    )
    sql = "\n".join(statements)

    assert statements[0] == "create schema if not exists gateway_usage;"
    for table in (
        "usage_events_staging",
        "usage_events",
        "usage_event_debug",
        "usage_agg_5m",
        "usage_agg_1h",
        "usage_agg_1d",
        "usage_dimension_values",
        "analytics_dimensions",
    ):
        assert f"gateway_usage.{table}" in sql
    assert "load_batches" not in sql
    assert "export_control" not in sql
    assert "accept_export_fence" not in sql


def test_analytics_dimensions_seed_rows_are_valid_sql_and_use_serving_columns():
    seed_sql = "\n".join(redshift_schema.analytics_dimensions_seed_statements())

    assert "delete from gateway_usage.analytics_dimensions;" in seed_sql
    assert "truncate" not in seed_sql.lower()
    for dimension in redshift_schema.ANALYTICS_DIMENSIONS:
        assert redshift_schema.sql_literal(dimension.dimension_kind) in seed_sql
        assert redshift_schema.sql_literal(dimension.source_column) in seed_sql
        assert redshift_schema.sql_literal(dimension.display_name) in seed_sql
    provider_model = next(
        dimension
        for dimension in redshift_schema.ANALYTICS_DIMENSIONS
        if dimension.dimension_kind == "provider_model"
    )
    assert provider_model.source_column == "provider_model"
    assert provider_model.fact_column == "requested_model_key"


def test_sql_literal_escapes_apostrophes_for_redshift_strings():
    assert redshift_schema.sql_literal("O'Reilly") == "'O''Reilly'"


def test_sql_builders_reject_unsafe_identifiers_and_expression_overrides():
    with pytest.raises(ValueError, match="SQL identifier"):
        redshift_schema.create_schema_ddl(
            "gateway_usage; drop schema public cascade; --"
        )

    with pytest.raises(ValueError, match="SQL identifier"):
        redshift_schema.analytics_dimensions_seed_statements(
            "gateway_usage; delete from x; --"
        )

    with pytest.raises(ValueError, match="SQL expression"):
        redshift_schema.aggregate_delete_sql(
            "5m", start_parameter=":start; drop table x"
        )

    with pytest.raises(ValueError, match="SQL expression"):
        redshift_schema.aggregate_insert_sql(
            "5m", data_through_expression=":data_through_utc; drop table x"
        )


def test_aggregate_refresh_sql_uses_missing_sentinel_and_excludes_non_additive_stats():
    delete_sql = redshift_schema.aggregate_delete_sql("5m")
    insert_sql = redshift_schema.aggregate_insert_sql("5m")

    assert "delete from gateway_usage.usage_agg_5m" in delete_sql
    assert "bucket_start_utc >= :start_bucket" in delete_sql
    assert "bucket_start_utc < :end_bucket" in delete_sql

    assert "insert into gateway_usage.usage_agg_5m" in insert_sql
    assert (
        "dateadd(minute, (floor(date_part(minute, completed_at::timestamp) / 5) * 5)::integer"
        in insert_sql
    )
    assert "coalesce(provider, '__missing__') as provider" in insert_sql
    assert (
        "coalesce(requested_model_key, '__missing__') as provider_model" in insert_sql
    )
    assert "coalesce(benchmark_name, '__missing__') as benchmark_name" in insert_sql
    assert "coalesce(agent_name, '__missing__') as agent_name" in insert_sql
    assert "coalesce(identity_email, '__missing__') as identity_email" in insert_sql
    assert (
        "coalesce(api_key_fingerprint, '__missing__') as api_key_fingerprint"
        in insert_sql
    )
    assert "reserved_dimension_guard" in insert_sql
    assert "then 0 else 1 end as reserved_dimension_guard" in insert_sql
    for dimension in redshift_schema.ANALYTICS_DIMENSIONS:
        if dimension.is_in_fast_agg:
            assert (
                f"{dimension.fact_column} in ({redshift_schema.sql_literal(redshift_schema.MISSING_DIMENSION_VALUE)})"
                in insert_sql
            )
    assert "count(*) as request_count" in insert_sql
    assert "count(duration_seconds) as duration_observed_count" in insert_sql
    assert (
        "group by\n  bucket_start_utc,\n  bucket_end_utc,\n  provider,\n  provider_model,"
        in insert_sql
    )
    assert "agent_name,\n  identity_email,\n  api_key_fingerprint;" in insert_sql
    assert "percentile_cont" not in insert_sql.lower()
    assert "count(distinct" not in insert_sql.lower()


def test_reserved_dimension_check_sql_counts_real_reserved_value_conflicts():
    check_sql = redshift_schema.aggregate_reserved_dimension_check_sql()

    assert "reserved_dimension_conflict_count" in check_sql
    assert "from gateway_usage.usage_events" in check_sql
    fast_fact_columns = [
        dimension.fact_column
        for dimension in redshift_schema.ANALYTICS_DIMENSIONS
        if dimension.is_in_fast_agg
    ]
    for fact_column in fast_fact_columns:
        assert (
            f"{fact_column} in ({redshift_schema.sql_literal(redshift_schema.MISSING_DIMENSION_VALUE)})"
            in check_sql
        )
    assert check_sql.count("  and (") == 1
    assert check_sql.rstrip().endswith(");")


def test_aggregate_refresh_sql_covers_hour_and_day_bucket_branches():
    hour_sql = redshift_schema.aggregate_insert_sql("1h")
    day_sql = redshift_schema.aggregate_insert_sql("1d")

    assert "insert into gateway_usage.usage_agg_1h" in hour_sql
    assert "date_trunc('hour', completed_at::timestamp) as bucket_start_utc" in hour_sql
    assert "dateadd(hour, 1, date_trunc('hour', completed_at::timestamp))" in hour_sql

    assert "insert into gateway_usage.usage_agg_1d" in day_sql
    assert "date_trunc('day', completed_at::timestamp) as bucket_start_utc" in day_sql
    assert "dateadd(day, 1, date_trunc('day', completed_at::timestamp))" in day_sql
