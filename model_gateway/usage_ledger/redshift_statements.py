"""Shared Redshift Data API statement completion."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Protocol


class DescribeStatementClient(Protocol):
    def describe_statement(self, **kwargs: object) -> Mapping[str, object]: ...


class RedshiftStatementError(RuntimeError):
    pass


def wait_for_statement(client: DescribeStatementClient, statement_id: str) -> None:
    while True:
        response = client.describe_statement(Id=statement_id)
        status = str(response["Status"])
        if status == "FINISHED":
            return
        if status in {"FAILED", "ABORTED"}:
            raise RedshiftStatementError(
                str(response.get("Error", "Redshift statement failed"))
            )
        time.sleep(1)
