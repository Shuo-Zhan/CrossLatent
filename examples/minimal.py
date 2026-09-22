"""Run after installing the core package: python examples/minimal.py --output demo."""
import argparse

from crosslatent.cli import demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(demo(args.output))
