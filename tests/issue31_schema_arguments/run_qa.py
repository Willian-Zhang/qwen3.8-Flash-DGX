#!/usr/bin/env python3
"""CPU schema-port QA against installed code; no model inference or tool execution."""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / "issue31_safe_output"
KNOWN_NAMESPACE_FAILURE = (
    "test_argument_schema.ArgumentSchema.test_namespace_uses_exact_offered_schema"
)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--report", type=Path, help="Optional writable JSON report path")
    cli.add_argument("--manifest", type=Path, default=ROOT / "PORT-MANIFEST.json",
                     help="Installed-source manifest for the tested vLLM release")
    args = cli.parse_args()
    # Only fixtures enter sys.path. vLLM must come from the installed image.
    sys.path[:0] = [str(ROOT), str(LEGACY), str(LEGACY / "tests")]
    package_root = Path(importlib.util.find_spec("vllm").origin).parent.parent
    manifest = json.loads(args.manifest.read_text())
    for entry in manifest["installed_files"]:
        path = package_root / entry["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry["target_sha256"]:
            raise SystemExit(f"Installed source hash mismatch: {entry['path']}")

    report = {"model_inference": False, "source_overlay": False,
              "installed_hashes_verified": len(manifest["installed_files"]),
              "groups": {}}
    passed = True
    for label, module, expected in (
        ("author_schema", "test_argument_schema", [KNOWN_NAMESPACE_FAILURE]),
        ("v031_integration", "schema_integration_tests", []),
        ("chat_compatibility", "chat_compat_tests", []),
    ):
        suite = unittest.defaultTestLoader.loadTestsFromModule(importlib.import_module(module))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        failures = [test.id() for test, _ in result.failures]
        errors = [test.id() for test, _ in result.errors]
        # Retain raw assertions and failures. A newly passing known case also
        # requires review of the documented v0.31 namespace limitation.
        matches = failures == expected and not errors and not result.skipped
        report["groups"][label] = {
            "methods": result.testsRun, "failures": failures, "errors": errors,
            "skipped": len(result.skipped), "expected_preexisting_failures": expected,
            "matches_expected": matches,
        }
        passed = passed and matches
    report["no_unexpected_failures"] = passed
    print(json.dumps(report, indent=2), flush=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
