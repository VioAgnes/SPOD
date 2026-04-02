#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SPOD Unified Entry Point
======================
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="SPOD: Semantic Prior-Orthogonal Decomposition")
    parser.add_argument("command", choices=["theory", "evaluate"])
    parser.add_argument("--subset", type=str, default="all")
    parser.add_argument("--mode", type=str, default="spod")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--test-mode", action="store_true")
    args, unknown = parser.parse_known_args()

    if args.command == "theory":
        from src.experiments.theory.analyze import main as theory_main
        sys.argv = ["analyze.py"] + sys.argv[2:]
        theory_main()
    else:
        from src.experiments.downstream.evaluate import main as eval_main
        sys.argv = ["evaluate.py"] + sys.argv[2:]
        eval_main()


if __name__ == "__main__":
    main()
