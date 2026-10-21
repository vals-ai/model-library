"""The AppConfig publisher behind PUT /admin/key-pools."""

from collections.abc import Mapping
from typing import Any

import pytest
from botocore.exceptions import ClientError

from model_gateway.key_pool_config import (
    AppConfigPoolConfigPublisher,
    PoolConfigDeploymentInProgressError,
    pool_config_publisher_from_env,
)


class FakeAppConfig:
    def __init__(self, *, deployment_error: str | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._deployment_error = deployment_error

    def create_hosted_configuration_version(self, **kwargs: Any) -> Mapping[str, Any]:
        self.calls.append(("create_hosted_configuration_version", kwargs))
        return {"VersionNumber": 4}

    def start_deployment(self, **kwargs: Any) -> Mapping[str, Any]:
        self.calls.append(("start_deployment", kwargs))
        if self._deployment_error is not None:
            raise ClientError(
                {"Error": {"Code": self._deployment_error, "Message": "busy"}},
                "StartDeployment",
            )
        return {"DeploymentNumber": 9, "State": "DEPLOYING"}


def _publisher(client: FakeAppConfig) -> AppConfigPoolConfigPublisher:
    return AppConfigPoolConfigPublisher(
        application_id="app",
        environment_id="env",
        profile_id="pools",
        deployment_strategy_id="immediate",
        client=client,
    )


async def test_publish_creates_a_version_and_deploys_it() -> None:
    client = FakeAppConfig()

    version = await _publisher(client).publish(b'{"pools":{}}')

    assert version == "4"
    (create, created), (deploy, deployed) = client.calls
    assert create == "create_hosted_configuration_version"
    assert created["ConfigurationProfileId"] == "pools"
    assert created["Content"] == b'{"pools":{}}'
    assert deploy == "start_deployment"
    assert deployed["ConfigurationVersion"] == "4"
    assert deployed["DeploymentStrategyId"] == "immediate"


async def test_publish_reports_a_deployment_already_in_progress() -> None:
    client = FakeAppConfig(deployment_error="ConflictException")

    with pytest.raises(PoolConfigDeploymentInProgressError):
        await _publisher(client).publish(b"{}")


_PUBLISHER_ENV = {
    "GATEWAY_APPCONFIG_APPLICATION": "gateway",
    "GATEWAY_APPCONFIG_ENVIRONMENT": "runtime",
    "GATEWAY_APPCONFIG_POOL_PROFILE": "pools",
    "GATEWAY_APPCONFIG_DEPLOYMENT_STRATEGY": "strategy",
}


def test_publisher_is_absent_without_an_appconfig_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _PUBLISHER_ENV:
        monkeypatch.delenv(name, raising=False)

    assert pool_config_publisher_from_env() is None


@pytest.mark.parametrize("missing", sorted(set(_PUBLISHER_ENV) - {"GATEWAY_APPCONFIG_APPLICATION"}))
def test_publisher_requires_every_setting_once_appconfig_is_on(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    for name, value in _PUBLISHER_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(missing)

    with pytest.raises(RuntimeError, match=missing):
        pool_config_publisher_from_env()
