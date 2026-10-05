"""Verify the exact file allowlist and SHA-256 hashes of a public release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath


IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "venv",
}
IGNORED_FILE_SUFFIXES = {".pyc", ".pyo"}
FORBIDDEN_SUFFIXES = {".csv", ".gz", ".parquet", ".npz", ".pt", ".pth", ".ckpt", ".joblib", ".pkl", ".pickle", ".log", ".pid"}
RELEASE_FLAGS = (
    "raw_market_data_included",
    "trained_weights_included",
    "timestamp_level_derivatives_included",
    "jao_authorization_included",
)


def is_runtime_metadata(path: Path, root: Path, *, strict: bool = False) -> bool:
    """Return true only for conventional VCS, environment, or build output."""

    relative = path.relative_to(root)
    if relative.parts[0] == ".git":
        return True
    if strict:
        return False
    return (
        any(
            (part in IGNORED_DIRECTORY_NAMES and part != "__pycache__")
            or part.endswith(".egg-info")
            for part in relative.parts[:-1]
        )
        or (
            "__pycache__" in relative.parts[:-1]
            and path.suffix.lower() in IGNORED_FILE_SUFFIXES
        )
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_release(root: Path, *, strict: bool = False) -> dict:
    """Validate the code-only contract, safe paths, allowlist, and content hashes."""

    root = root.expanduser().resolve()
    manifest_path = root / "release_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing release manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("release_type") != "code_only"
        or any(manifest.get(flag) is not False for flag in RELEASE_FLAGS)
        or not isinstance(manifest.get("files"), list)
        or not manifest["files"]
    ):
        raise RuntimeError("Invalid code-only release manifest metadata")
    expected = {}
    for entry in manifest["files"]:
        relative = entry.get("path")
        if not isinstance(relative, str):
            raise RuntimeError("Manifest path must be a string")
        parts = PurePosixPath(relative).parts
        if (
            not parts
            or PurePosixPath(relative).is_absolute()
            or any(part in {".", ".."} for part in parts)
            or "\\" in relative
            or ":" in relative
            or PurePosixPath(relative).as_posix() != relative
            or relative == "release_manifest.json"
            or parts[0] == ".git"
            or PurePosixPath(relative).suffix.lower() in FORBIDDEN_SUFFIXES
        ):
            raise RuntimeError(f"Unsafe or non-code manifest path: {relative}")
        if (
            relative in expected
            or not isinstance(entry.get("bytes"), int)
            or isinstance(entry.get("bytes"), bool)
            or entry["bytes"] < 0
            or not isinstance(entry.get("sha256"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"])
        ):
            raise RuntimeError(f"Invalid or duplicate manifest entry: {relative}")
        expected[relative] = entry

    for path in root.rglob("*"):
        if path.is_symlink() and not is_runtime_metadata(path, root, strict=strict):
            raise RuntimeError(f"Release contains a symbolic link: {path.relative_to(root)}")
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
        and path != manifest_path
        and not is_runtime_metadata(path, root, strict=strict)
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

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Public-release root (defaults to the parent of scripts/).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Reject runtime/build artifacts too; only Git metadata is ignored.",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    manifest = verify_release(root, strict=args.strict)
    print(f"[OK] release manifest verified: files={len(manifest['files'])} root={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
