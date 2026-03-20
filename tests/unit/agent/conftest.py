import pytest

from model_library.agent.agent import Agent


@pytest.fixture(autouse=True)
def agent_tmp_log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        Agent,
        "_build_log_dir",
        staticmethod(lambda base, name, model_name: tmp_path / name / model_name / "run"),
    )
