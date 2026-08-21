from pathlib import Path
import runpy

import pytest


def test_validate_result_logs_and_raises_on_mismatch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts" / "verify_gpu_stack.py")
    )
    validate_result = script.get("_validate_result")
    assert validate_result is not None, "smoke results must be validated"

    with pytest.raises(
        RuntimeError,
        match=r"Triton smoke failed: expected=42, actual=41",
    ):
        validate_result("Triton", 41.0)

    assert "Triton smoke failed: expected=42, actual=41" in capsys.readouterr().err


def test_validate_result_logs_success(capsys: pytest.CaptureFixture[str]) -> None:
    script = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts" / "verify_gpu_stack.py")
    )
    validate_result = script["_validate_result"]

    validate_result("CuTe", 42.0)

    assert "CuTe smoke passed: result=42" in capsys.readouterr().err
