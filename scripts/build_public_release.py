"""Build and verify a code-only AF-CDMO public-release directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "public_release" / "af_cdmo_code_only"

ROOT_FILES = {
    "README_PUBLIC.md": "README.md",
    "LICENSE": "LICENSE",
    "CITATION.cff": "CITATION.cff",
    "CONTRIBUTING.md": "CONTRIBUTING.md",
    "SECURITY.md": "SECURITY.md",
    "DATA_ACCESS_AND_RELEASE.md": "DATA_ACCESS_AND_RELEASE.md",
    "pyproject.toml": "pyproject.toml",
    "requirements.txt": "requirements.txt",
    "pytest.ini": "pytest.ini",
    ".github/workflows/tests.yml": ".github/workflows/tests.yml",
}

SCRIPT_FILES = (
    "build_public_release.py",
    "af_cdmo_real_controls_v14.py",
    "af_cdmo_real_extension_v13.py",
    "audit_dual_measure_conservation.py",
    "audit_af_cdmo_real_geometry_v13.py",
    "audit_af_cdmo_real_v14.py",
    "analyze_af_cdmo_real_v14_calendar_bootstrap.py",
    "benchmark_af_cdmo_real_v13.py",
    "benchmark_af_cdmo_real_v14.py",
    "benchmark_certificate_governance_v5.py",
    "benchmark_multizone_quotient_gauge_v6.py",
    "benchmark_multizone_factorial_controls_v10.py",
    "cqdm_utils.py",
    "multizone_factorial_controls_v10.py",
    "multizone_quotient_gauge_v6.py",
    "run_af_cdmo_real_v14.ps1",
    "slack_qdm_extension_v11.py",
    "verify_release_manifest.py",
)

TEST_FILES = (
    "test_af_cdmo_theorem_v12.py",
    "test_af_cdmo_extension_v12.py",
    "test_af_cdmo_real_geometry_v13.py",
    "test_af_cdmo_real_extension_v13.py",
    "test_benchmark_af_cdmo_real_v13.py",
    "test_af_cdmo_real_controls_v14.py",
    "test_af_cdmo_real_calendar_bootstrap_v14.py",
)

FORBIDDEN_SUFFIXES = {
    ".csv",
    ".gz",
    ".parquet",
    ".npz",
    ".pt",
    ".pth",
    ".ckpt",
    ".log",
    ".pid",
}
ABSOLUTE_LOCAL_PATH = re.compile(r"(?i)[A-Z]:\\Users\\")
SECRET_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|securitytoken|token|secret|password)\s*[:=]\s*"
    r"[\"'](?!<|YOUR_|REDACTED)[A-Za-z0-9_./+=-]{16,}[\"']"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Missing release source: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    release_root = (ROOT / "public_release").resolve()
    output = args.output.expanduser().resolve()
    if output == release_root or release_root not in output.parents:
        raise ValueError(f"Output must be a child of {release_root}")
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Refusing to overwrite {output}; use --force")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for source_name, destination_name in ROOT_FILES.items():
        copy_file(ROOT / source_name, output / destination_name)
    for source in sorted((ROOT / "af_cdmo").glob("*.py")):
        copy_file(source, output / "af_cdmo" / source.name)
    for name in SCRIPT_FILES:
        copy_file(ROOT / "scripts" / name, output / "scripts" / name)
    for name in TEST_FILES:
        copy_file(ROOT / "tests" / name, output / "tests" / name)

    (output / ".gitignore").write_text(
        "\n".join(
            [
                "__pycache__/",
                "*.py[cod]",
                ".pytest_cache/",
                ".env",
                ".env.*",
                "official_jao_dual_measure/",
                "official_entsoe_nordic_day_ahead/",
                "results/",
                "**/*.pt",
                "**/*.npz",
                "**/*.log",
                "",
            ]
        ),
        encoding="utf-8",
    )

    violations: list[str] = []
    files = sorted(path for path in output.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(output).as_posix()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append(f"forbidden extension: {relative}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            violations.append(f"unexpected binary file: {relative}")
            continue
        if ABSOLUTE_LOCAL_PATH.search(text):
            violations.append(f"absolute local path: {relative}")
        if SECRET_PATTERN.search(text):
            violations.append(f"possible embedded credential: {relative}")
    if violations:
        shutil.rmtree(output)
        raise RuntimeError("Public-release audit failed:\n  " + "\n  ".join(violations))

    manifest_files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        manifest_files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "release_type": "code_only",
        "raw_market_data_included": False,
        "trained_weights_included": False,
        "timestamp_level_derivatives_included": False,
        "jao_authorization_included": False,
        "files": manifest_files,
    }
    manifest_path = output / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] code-only release: {output}")
    print(f"[OK] files={len(manifest_files)} manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
