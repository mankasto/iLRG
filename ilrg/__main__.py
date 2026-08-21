"""Command-line entry point for ``python -m ilrg``."""

from runpy import run_module


if __name__ == "__main__":
    run_module("ilrg.main", run_name="__main__")
