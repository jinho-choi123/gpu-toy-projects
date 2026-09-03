"""Verify the profiling sweep's user-visible dry-run contract."""

from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SWEEP_SCRIPT = PROJECT_DIR / "scripts" / "sweep_profiles.sh"


def _dry_run() -> str:
    return subprocess.run(
        [SWEEP_SCRIPT, "--dry-run", "--force"],
        cwd=PROJECT_DIR,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_dry_run_schedules_every_qwen_profile() -> None:
    """Schedule every backend, sequence length, and profiler combination."""
    output = _dry_run()
    commands = [
        line
        for line in output.splitlines()
        if line.startswith("  command:") and "profile_qwen38_27b_" in line
    ]

    assert len(commands) == 12
    assert "Profiling sweep: 48 jobs (36 GDN + 12 Qwen3.8-27B)" in output

    for backend in ("flash_qla", "fla_triton"):
        for seq_len in (16384, 32768, 65536):
            matching_commands = [
                command
                for command in commands
                if f"profile_qwen38_27b_{backend}.py" in command
                and f"--seq-len {seq_len}" in command
            ]
            assert len(matching_commands) == 2
            assert any(
                "nsys profile" in command and f"qwen38_27b_{backend}_b1_t{seq_len}_nsys" in command
                for command in matching_commands
            )
            assert any(
                "ncu --section SpeedOfLight --replay-mode app-range" in command
                and f"qwen38_27b_{backend}_b1_t{seq_len}_ncu" in command
                for command in matching_commands
            )


def test_dry_run_preserves_qwen_memory_snapshots_per_profiler() -> None:
    """Keep each profiler's memory record and use workload-specific options."""
    output = _dry_run()
    qwen_commands = [
        line
        for line in output.splitlines()
        if line.startswith("  command:") and "profile_qwen38_27b_" in line
    ]
    qwen_snapshots = [
        line
        for line in output.splitlines()
        if line.startswith("  snapshot:") and "qwen38_27b_" in line
    ]
    nsys_commands = [line for line in qwen_commands if "nsys profile" in line]
    ncu_commands = [
        line
        for line in qwen_commands
        if "ncu --section SpeedOfLight --replay-mode app-range" in line
    ]

    assert len(nsys_commands) == 6
    assert len(ncu_commands) == 6
    assert len(qwen_snapshots) == 12
    assert sum("_nsys_memory_snapshot.pickle" in line for line in qwen_snapshots) == 6
    assert sum("_ncu_memory_snapshot.pickle" in line for line in qwen_snapshots) == 6
    assert all("--capture-range=cudaProfilerApi" in command for command in nsys_commands)
    assert all("regex:qwen38_gdn_decoder_layer_" in command for command in ncu_commands)
    assert all("--profile-from-start" not in command for command in ncu_commands)
    assert all("--graph-profiling" not in command for command in ncu_commands)
