from tempfile import NamedTemporaryFile
from typing_extensions import TypedDict
from ruff.__main__ import find_ruff_bin
import subprocess


class LintOutput(TypedDict):
    out: str
    error: str

def run_ruff(code: str) -> LintOutput:
    with NamedTemporaryFile(mode="w", suffix=".py") as f:
        f.write(code)
        f.seek(0)
        ruff_binary = find_ruff_bin()
        res = subprocess.run([ruff_binary, f.name], capture_output=True)
        output, err = res.stdout, res.stderr
        # Replace the temp file name
        result = output.decode().replace(f.name, "code.py")
        error = err.decode().replace(f.name, "code.py")
        return {
            "out": result,
            "error": error,
        }
