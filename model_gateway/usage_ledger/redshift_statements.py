"""Shared Redshift Data API statement completion."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Mapping
from typing import Protocol

from botocore.exceptions import BotoCoreError, ClientError


class DescribeStatementClient(Protocol):
    def describe_statement(self, **kwargs: object) -> Mapping[str, object]: ...


class CancelStatementClient(DescribeStatementClient, Protocol):
    def cancel_statement(self, **kwargs: object) -> Mapping[str, object]: ...


class RedshiftStatementError(RuntimeError):
    pass


class RedshiftStatementTimeout(RedshiftStatementError):
    pass


def wait_for_statement(
    client: DescribeStatementClient, statement_id: str, deadline: float | None = None
) -> None:
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise RedshiftStatementTimeout("Redshift statement timed out")
        response = client.describe_statement(Id=statement_id)
        status = str(response["Status"])
        if status == "FINISHED":
            return
        if status in {"FAILED", "ABORTED"}:
            raise RedshiftStatementError(
                str(response.get("Error", "Redshift statement failed"))
            )
        time.sleep(1)


def wait_for_statement_within(
    client: CancelStatementClient, statement_id: str, seconds: float
) -> None:
    """Wait at most `seconds`, then cancel the statement so Redshift stops running it."""
    try:
        wait_for_statement(client, statement_id, time.monotonic() + seconds)
    except RedshiftStatementTimeout:
        with contextlib.suppress(BotoCoreError, ClientError):
            client.cancel_statement(Id=statement_id)
        raise
