"""Check payload identity and optionally reject private strings in public files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_SHA256 = "c911d8146a0ca02a31ec80595b13611c94f097af21d60525bc08f447c446d50b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forbid", action="append", default=[])
    args = parser.parse_args()
    manifest = json.loads((ROOT / "release-manifest.json").read_text())
    errors = []
    for name, expected in manifest["files"].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"payload mismatch: {name}")
    engine = ROOT / "benchmark_harness/reference_evaluation/engine.py"
    if hashlib.sha256(engine.read_bytes()).hexdigest() != EVALUATOR_SHA256:
        errors.append("frozen evaluator mismatch")
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True
    ).splitlines()
    actual = set(names) - {"release-manifest.json"}
    if actual != set(manifest["files"]):
        errors.append("public file inventory differs from release manifest")
    for name in set(names):
        path = ROOT / name
        if not path.is_file():
            continue
        value = path.read_bytes()
        if any(term.encode().lower() in value.lower() for term in args.forbid):
            errors.append(f"private reference: {name}")
        if re.search(rb"/(?:Users|home)/[^/\s]+/", value):
            errors.append(f"developer path: {name}")
        if re.search(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----", value):
            errors.append(f"private key: {name}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Release verified: {len(manifest['files'])} payloads; evaluator identity intact")


if __name__ == "__main__":
    main()
