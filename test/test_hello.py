"""Tests for ``src/hello.py``."""

import subprocess
import sys
from pathlib import Path


def test_hello_runs():
    """Running hello.py exits cleanly and prints the greeting."""
    script = Path(__file__).resolve().parent.parent / "src" / "hello.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Hello world" in result.stdout
