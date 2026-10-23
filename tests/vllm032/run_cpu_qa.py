#!/usr/bin/env python3
"""Run v0.32 installed-code CPU checks, preserving each suite's diagnostics."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, help="Optional writable report directory")
    args = cli.parse_args()
    if not os.environ.get("QWEN_TOKENIZER"):
        cli.error("set QWEN_TOKENIZER to local compatible tokenizer metadata")
    tests = Path(__file__).resolve().parents[1]
    suites = {
        "schema": ["issue31_schema_arguments/run_qa.py", "--manifest",
                   str(tests / "vllm032/PORT-MANIFEST.json")],
        "safe-output": ["issue31_safe_output/run_qa.py"],
        "mtp-aux": ["test_mtp_aux_output_cpu.py"],
        "hybrid-qsa": ["vllm032/test_hybrid_qsa.py"],
        "hybrid-dispatch": ["vllm032/test_hybrid_dispatch.py"],
        "ple-constructor": ["vllm032/test_ple_constructor.py"],
        "responses": ["vllm032/test_responses.py"],
        "mrope-window": ["vllm032/test_mrope_window.py"],
    }
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"model_inference": False, "suites": {}}
    for label, command in suites.items():
        result = subprocess.run(
            [sys.executable, str(tests / command[0]), *command[1:]],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            check=False,
        )
        print(f"\n=== {label} ===\n{result.stdout}", flush=True)
        report["suites"][label] = {"exit_code": result.returncode}
        if args.output_dir:
            (args.output_dir / f"{label}.txt").write_text(result.stdout)
    report["no_unexpected_failures"] = all(
        suite["exit_code"] == 0 for suite in report["suites"].values()
    )
    print(json.dumps(report, indent=2), flush=True)
    if args.output_dir:
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["no_unexpected_failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
