"""
Capture a snapshot of the runtime environment (Python version, installed
package versions, platform, git commit) into ENVIRONMENT.md. Run this
once at freeze time (Step 1) and again immediately before the live
GPT-5.4 evaluation (Step 2), so both snapshots are on record.

Usage:
    python3 -m evaluation.semantic_orchestration.provenance.capture_environment [--label STEP1|STEP2]
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

PACKAGES_OF_INTEREST = [
    "openai",
    "fastapi",
    "uvicorn",
    "pydantic",
    "gradio",
    "huggingface_hub",
    "requests",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "reportlab",
    "rpy2",
    "pytest",
]


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {e}>"


def _pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def build_report(label: str) -> str:
    lines = [
        f"# Environment snapshot ({label})",
        "",
        f"Captured: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Git",
        f"- commit: `{_git('rev-parse', 'HEAD')}`",
        f"- branch: `{_git('rev-parse', '--abbrev-ref', 'HEAD')}`",
        f"- dirty: `{_git('status', '--porcelain')!r}` (empty string means clean)",
        "",
        "## Interpreter",
        f"- python: `{sys.version}`",
        f"- executable: `{sys.executable}`",
        f"- platform: `{platform.platform()}`",
        f"- machine: `{platform.machine()}`",
        "",
        "## Package versions (installed vs. requirements.txt pin)",
        "",
        "| Package | Installed | Pinned in requirements.txt |",
        "|---|---|---|",
    ]

    pinned = _parse_requirements_pins()
    for pkg in PACKAGES_OF_INTEREST:
        lines.append(f"| {pkg} | {_pkg_version(pkg)} | {pinned.get(pkg, 'n/a')} |")

    lines += [
        "",
        "## Notes",
        (
            "- `openai` is intentionally excluded from the version-mismatch "
            "warning list below if it matches; see AUDIT.md Finding 6 for "
            "why the installed version can legitimately differ from the "
            "requirements.txt pin in an evaluation-only environment."
        ),
    ]

    mismatches = []
    for pkg in PACKAGES_OF_INTEREST:
        installed = _pkg_version(pkg)
        pin = pinned.get(pkg)
        if pin and installed not in ("NOT INSTALLED",) and installed != pin:
            mismatches.append(f"  - {pkg}: installed={installed}, pinned={pin}")
    if mismatches:
        lines.append("- Version mismatches vs. requirements.txt:")
        lines.extend(mismatches)
    else:
        lines.append("- No version mismatches detected among installed packages.")

    return "\n".join(lines) + "\n"


def _parse_requirements_pins() -> dict:
    req_path = REPO_ROOT / "requirements.txt"
    pins = {}
    if not req_path.exists():
        return pins
    for line in req_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, _, version = line.partition("==")
        name = name.split("[")[0].strip()
        pins[name] = version.strip()
    return pins


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="STEP1", help="Label for this snapshot, e.g. STEP1 or STEP2.")
    ap.add_argument("--out", default=str(HERE / "ENVIRONMENT.md"))
    args = ap.parse_args()

    report = build_report(args.label)
    out_path = Path(args.out)

    if out_path.exists():
        existing = out_path.read_text(encoding="utf-8")
        report = existing.rstrip() + "\n\n---\n\n" + report

    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
