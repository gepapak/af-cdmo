"""Verify the exact file allowlist and SHA-256 hashes of a public release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "venv",
}
IGNORED_FILE_SUFFIXES = {".pyc", ".pyo"}


def is_runtime_metadata(path: Path, root: Path) -> bool:
    """Return true only for conventional VCS, environment, or build output."""

    relative = path.relative_to(root)
    return (
        any(
            part in IGNORED_DIRECTORY_NAMES or part.endswith(".egg-info")
            for part in relative.parts[:-1]
        )
        or path.suffix.lower() in IGNORED_FILE_SUFFIXES
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Public-release root (defaults to the parent of scripts/).",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    manifest_path = root / "release_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing release manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {entry["path"]: entry for entry in manifest["files"]}
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
        and path != manifest_path
        and not is_runtime_metadata(path, root)
    }

    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatches: list[str] = []
    for relative in sorted(set(expected) & set(actual)):
        path = actual[relative]
        entry = expected[relative]
        digest = sha256_file(path)
        if path.stat().st_size != entry["bytes"] or digest != entry["sha256"]:
            mismatches.append(relative)

    if missing or unexpected or mismatches:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        if mismatches:
            details.append(f"hash_or_size_mismatch={mismatches}")
        raise RuntimeError("Release verification failed: " + "; ".join(details))

    print(f"[OK] release manifest verified: files={len(expected)} root={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
