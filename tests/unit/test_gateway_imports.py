"""Import isolation tests for gateway client modules."""

import os
import subprocess
import sys
from pathlib import Path


def test_model_gateway_types_imports_without_model_library_cycle():
    result = subprocess.run(
        [sys.executable, "-c", "import model_gateway.types"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_model_library_base_reexports_gateway_llm():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from model_library.base import GatewayLLM; print(GatewayLLM.__name__)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "GatewayLLM"


def test_model_library_import_does_not_depend_on_top_level_server_package(
    tmp_path: Path,
):
    shadow_server = tmp_path / "server"
    shadow_server.mkdir()
    (shadow_server / "__init__.py").write_text("", encoding="utf-8")
    (shadow_server / "types.py").write_text("SHADOW_TYPES = True\n", encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(tmp_path),
            str(repo_root),
            env.get("PYTHONPATH", ""),
        ]
    )

    result = subprocess.run(
        [sys.executable, "-c", "import model_library"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
