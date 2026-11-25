"""Command-line interface; heavyweight dependencies load only for execution."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="vidhalluc", description="Video hallucination inference and evaluation"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser("infer", help="Run SPOD inference")
    infer.add_argument("--config", default="configs/spod.json")
    infer.add_argument("--output-dir", default="outputs/default")
    infer.add_argument("--prompts-per-batch", type=int, default=1)
    infer.add_argument(
        "--smoke", action="store_true", help="Use eight inputs per subset"
    )
    infer.add_argument("--shard", type=int, default=0)
    infer.add_argument("--shards", type=int, default=1)
    evaluate = commands.add_parser("evaluate", help="Score generated predictions")
    evaluate.add_argument("directory")
    evaluate.add_argument("--config", default="configs/spod.json")
    evaluate.add_argument("--partial", action="store_true")
    audit = commands.add_parser("audit", help="Inspect answer-format sensitivity")
    audit.add_argument("subset", choices=("mcq", "tsh"))
    audit.add_argument("directory")
    args = parser.parse_args()
    if args.command == "infer":
        from .inference import main as run

        del args.command
        run(args)
    elif args.command == "evaluate":
        from .evaluation import main as run

        del args.command
        run(args)
    else:
        if args.subset == "mcq":
            from .diagnostics.audit_mcq_semantics import main as run
        else:
            from .diagnostics.audit_tsh_semantics import main as run
        sys.argv = [sys.argv[0], args.directory]
        run()
