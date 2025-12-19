import pytest

from model_library.base.output import QueryResultCost, QueryResultMetadata


@pytest.mark.unit
class TestQueryResultCostAddition:
    def test_add_full_costs(self):
        cost1 = QueryResultCost(
            input=0.01,
            output=0.02,
            reasoning=0.005,
            cache_read=0.001,
            cache_write=0.002,
        )
        cost2 = QueryResultCost(
            input=0.02,
            output=0.03,
            reasoning=0.01,
            cache_read=0.002,
            cache_write=0.003,
        )

        result = cost1 + cost2

        assert result.input == 0.03
        assert result.output == 0.05
        assert result.reasoning == 0.015
        assert result.cache_read == 0.003
        assert result.cache_write == 0.005

    def test_add_costs_with_none_fields(self):
        cost1 = QueryResultCost(input=0.01, output=0.02, reasoning=0.005)
        cost2 = QueryResultCost(input=0.02, output=0.03, cache_read=0.002)

        result = cost1 + cost2

        assert result.input == 0.03
        assert result.output == 0.05
        assert result.reasoning == 0.005
        assert result.cache_read == 0.002
        assert result.cache_write is None

    def test_add_costs_both_none_fields(self):
        cost1 = QueryResultCost(input=0.01, output=0.02)
        cost2 = QueryResultCost(input=0.02, output=0.03)

        result = cost1 + cost2

        assert result.input == 0.03
        assert result.output == 0.05
        assert result.reasoning is None
        assert result.cache_read is None
        assert result.cache_write is None

    def test_cost_total_computed(self):
        cost1 = QueryResultCost(input=0.01, output=0.02, reasoning=0.005)
        cost2 = QueryResultCost(input=0.02, output=0.03, cache_read=0.001)

        result = cost1 + cost2

        expected_total = 0.03 + 0.05 + 0.005 + 0.001
        assert abs(result.total - expected_total) < 1e-10


@pytest.mark.unit
class TestQueryResultMetadataAddition:
    def test_add_full_metadata(self):
        meta1 = QueryResultMetadata(
            in_tokens=100,
            out_tokens=50,
            reasoning_tokens=20,
            cache_read_tokens=10,
            cache_write_tokens=5,
            duration_seconds=1.5,
            cost=QueryResultCost(input=0.01, output=0.02),
        )
        meta2 = QueryResultMetadata(
            in_tokens=200,
            out_tokens=100,
            reasoning_tokens=30,
            cache_read_tokens=15,
            cache_write_tokens=10,
            duration_seconds=2.0,
            cost=QueryResultCost(input=0.02, output=0.03),
        )

        result = meta1 + meta2

        assert result.in_tokens == 300
        assert result.out_tokens == 150
        assert result.reasoning_tokens == 50
        assert result.cache_read_tokens == 25
        assert result.cache_write_tokens == 15
        assert result.duration_seconds == 3.5
        assert result.cost is not None
        assert result.cost.input == 0.03
        assert result.cost.output == 0.05

    def test_add_metadata_missing_optional_tokens(self):
        meta1 = QueryResultMetadata(
            in_tokens=100,
            out_tokens=50,
            reasoning_tokens=20,
            duration_seconds=1.0,
        )
        meta2 = QueryResultMetadata(
            in_tokens=200,
            out_tokens=100,
            cache_read_tokens=15,
            duration_seconds=2.0,
        )

        result = meta1 + meta2

        assert result.in_tokens == 300
        assert result.out_tokens == 150
        assert result.reasoning_tokens == 20
        assert result.cache_read_tokens == 15
        assert result.cache_write_tokens is None

    def test_add_metadata_both_missing_optional_tokens(self):
        meta1 = QueryResultMetadata(in_tokens=100, out_tokens=50)
        meta2 = QueryResultMetadata(in_tokens=200, out_tokens=100)

        result = meta1 + meta2

        assert result.in_tokens == 300
        assert result.out_tokens == 150
        assert result.reasoning_tokens is None
        assert result.cache_read_tokens is None
        assert result.cache_write_tokens is None

    def test_add_metadata_one_has_cost(self):
        meta1 = QueryResultMetadata(
            in_tokens=100,
            out_tokens=50,
            cost=QueryResultCost(input=0.01, output=0.02),
        )
        meta2 = QueryResultMetadata(in_tokens=200, out_tokens=100, cost=None)

        result = meta1 + meta2

        assert result.cost is not None
        assert result.cost.input == 0.01
        assert result.cost.output == 0.02

    def test_add_metadata_other_has_cost(self):
        meta1 = QueryResultMetadata(in_tokens=100, out_tokens=50, cost=None)
        meta2 = QueryResultMetadata(
            in_tokens=200,
            out_tokens=100,
            cost=QueryResultCost(input=0.02, output=0.03),
        )

        result = meta1 + meta2

        assert result.cost is not None
        assert result.cost.input == 0.02
        assert result.cost.output == 0.03

    def test_add_metadata_neither_has_cost(self):
        meta1 = QueryResultMetadata(in_tokens=100, out_tokens=50)
        meta2 = QueryResultMetadata(in_tokens=200, out_tokens=100)

        result = meta1 + meta2

        assert result.cost is None

    def test_add_metadata_default_duration(self):
        meta1 = QueryResultMetadata(in_tokens=100, out_tokens=50, duration_seconds=None)
        meta2 = QueryResultMetadata(in_tokens=200, out_tokens=100, duration_seconds=2.0)

        result = meta1 + meta2

        assert result.duration_seconds == 2.0

    def test_add_metadata_both_none_duration(self):
        meta1 = QueryResultMetadata(in_tokens=100, out_tokens=50, duration_seconds=None)
        meta2 = QueryResultMetadata(
            in_tokens=200, out_tokens=100, duration_seconds=None
        )

        result = meta1 + meta2

        assert result.duration_seconds == 0.0

    def test_computed_total_tokens(self):
        meta1 = QueryResultMetadata(
            in_tokens=100,
            out_tokens=50,
            reasoning_tokens=20,
            cache_read_tokens=10,
            cache_write_tokens=5,
        )
        meta2 = QueryResultMetadata(
            in_tokens=200,
            out_tokens=100,
            reasoning_tokens=30,
            cache_read_tokens=15,
        )

        result = meta1 + meta2

        assert result.total_input_tokens == 300 + 25 + 5
        assert result.total_output_tokens == 150 + 50
