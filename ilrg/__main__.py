"""Command-line entry point for ``python -m ilrg``."""

from .options import options


def main():
    args = options().parse_args()
    # Import the numerical stack only after parsing, so that ``python -m ilrg --help``
    # works before the research dependencies are installed.
    from .experiment import run_experiments
    run_experiments(args)


if __name__ == "__main__":
    main()
