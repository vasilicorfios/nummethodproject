import subprocess
import sys
from pathlib import Path


def test_hello_runs():
    script = Path(__file__).resolve().parent.parent / "src" / "hello.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Hello world" in result.stdout
