"""Tests for atlas-ai __main__ entry point."""

import subprocess
import sys


def _run(*args, stdin_data=None):
    """Run atlas_ai module with given args, return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-m", "atlas_ai", *args],
        capture_output=True,
        text=True,
        input=stdin_data,
        timeout=15,
    )
    return result.returncode, result.stdout, result.stderr


class TestMain:
    def test_info_exits_zero(self):
        code, _, _ = _run("--info")
        assert code == 0

    def test_info_prints_provider(self):
        _, out, _ = _run("--info")
        assert "Provider:" in out

    def test_info_prints_model(self):
        _, out, _ = _run("--info")
        assert "Model:" in out

    def test_info_prints_base_url(self):
        _, out, _ = _run("--info")
        assert "Base URL:" in out

    def test_info_prints_mode(self):
        _, out, _ = _run("--info")
        assert "Mode:" in out

    def test_info_default_mode_is_stdin(self):
        _, out, _ = _run("--info")
        assert "stdin" in out

    def test_help_exits_zero(self):
        code, _, _ = _run("--help")
        assert code == 0

    def test_stdin_mode_invalid_json_exits_nonzero(self):
        code, _, _ = _run("--mode", "stdin", stdin_data="not valid json")
        assert code != 0
