"""Run the paper notebooks and save their figures inline."""

import argparse
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def render(path):
    original_bytes = path.read_bytes()
    notebook = nbformat.read(path, as_version=4)
    notebook.cells.insert(0, nbformat.v4.new_code_cell("%matplotlib inline"))
    # Run the notebook kernel with this Python interpreter.
    kernel = KernelManager(kernel_name="python3")
    kernel.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    try:
        NotebookClient(notebook, km=kernel, timeout=1200,
                       resources={"metadata": {"path": str(path.parent)}}).execute()
    finally:
        if kernel.has_kernel:
            kernel.shutdown_kernel(now=True)
    notebook.cells.pop(0)
    if not any("image/png" in output.get("data", {})
               for cell in notebook.cells if cell.cell_type == "code"
               for output in cell.get("outputs", [])):
        raise RuntimeError(f"No inline figure produced: {path.name}")
    # Keep inline figures and clear transient execution output.
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = [output for output in cell.outputs
                            if output.output_type != "stream"
                            and not (output.output_type == "execute_result"
                                     and "image/png" not in output.get("data", {}))]
    if path.read_bytes() != original_bytes:
        raise RuntimeError(f"Notebook was edited during execution; leaving those edits intact: {path.name}")
    nbformat.write(notebook, path)
    print(f"Rendered {path.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all == bool(args.notebooks):
        parser.error("choose --all or explicit notebook filenames")
    os.environ.setdefault("MPLBACKEND", "Agg")
    paths = sorted(NOTEBOOKS.glob("*.ipynb")) if args.all else [NOTEBOOKS / name for name in args.notebooks]
    for path in paths:
        render(path)


if __name__ == "__main__":
    main()
