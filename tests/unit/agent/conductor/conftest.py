import pytest

from model_library.agent.agent import Agent
from model_library.agent.conductor.conductor import ConductorAgent


@pytest.fixture(autouse=True)
def agent_tmp_log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        Agent,
        "_build_log_dir",
        staticmethod(lambda base, name, model_name: tmp_path / name / model_name / "run"),
    )


@pytest.fixture(autouse=True)
def conductor_tmp_log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ConductorAgent,
        "_build_log_dir",
        staticmethod(
            lambda base, name, auditor_model, target_model: tmp_path
            / name
            / f"{auditor_model}_{target_model}"
            / "run"
        ),
    )
