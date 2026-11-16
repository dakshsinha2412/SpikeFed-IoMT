"""Offline structural checks for SpikeFed-IoMT.

Run this on the PC before deploying to a Pi:
    python self_check.py

It intentionally does not download data or contact a Raspberry Pi.
"""

from __future__ import annotations

import ast
import pathlib
import py_compile
import sys


ROOT = pathlib.Path(__file__).resolve().parent
PYTHON_FILES = [
    "main.py", "client.py", "server.py", "model.py", "evaluate.py",
    "prepare_data.py", "utils.py", "dataset.py", "federated.py", "metrics.py", "dashboard.py", "tests.py",
]
REQUIRED_PI_FILES = [
    "client.py", "model.py", "metrics.py", "utils.py", "requirements_pi.txt",
    "pi_setup/pi_install.sh", "pi_setup/start_client.sh",
]


def main() -> int:
    failed = False
    for relative_path in PYTHON_FILES:
        path = ROOT / relative_path
        try:
            py_compile.compile(str(path), doraise=True)
            print(f"OK   Python syntax: {relative_path}")
        except py_compile.PyCompileError as exc:
            failed = True
            print(f"FAIL Python syntax: {relative_path}\n     {exc.msg}")

    for relative_path in REQUIRED_PI_FILES:
        path = ROOT / relative_path
        if path.is_file():
            print(f"OK   Pi deployment file: {relative_path}")
        else:
            failed = True
            print(f"FAIL Missing Pi deployment file: {relative_path}")

    # Verify that the server checkpoint path consumes aggregated parameters.
    server_tree = ast.parse((ROOT / "server.py").read_text(encoding="utf-8"))
    server_source = ast.unparse(server_tree)
    if "latest_parameters" in server_source and "parameters_to_ndarrays" in server_source:
        print("OK   Final checkpoint uses aggregated server parameters")
    else:
        failed = True
        print("FAIL Final checkpoint is not connected to aggregated parameters")

    print("\nSelf-check " + ("failed." if failed else "passed."))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
