"""Small, offline entry point for the project scaffold."""

import argparse

from shrek import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="shrek",
        description="SHREK: regression-tested defensive memory for frozen LLMs.",
        epilog="Scaffold only: research components are not implemented yet.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args()
    parser.print_help()


if __name__ == "__main__":
    main()
