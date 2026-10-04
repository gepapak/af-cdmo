"""Sync a verified code-only release to the GitHub folder and create its ZIP.

Only files registered in release manifests are copied or removed. Git metadata
and existing runtime caches are preserved, and the ZIP contains only allowlisted
files. No commit, push, or publication is performed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

try:
    from scripts.verify_release_manifest import is_runtime_metadata, verify_release
except ModuleNotFoundError:
    from verify_release_manifest import is_runtime_metadata, verify_release


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "public_release" / "af_cdmo_code_only"
DEFAULT_TARGET = ROOT / "GitHub" / "AF-CDMO"
DEFAULT_ARCHIVE = ROOT / "GitHub" / "AF-CDMO-code-only.zip"


def package_release(source: Path, target: Path, archive: Path) -> int:
    """Synchronize a clean release within the local GitHub staging directory."""

    source = source.expanduser().resolve()
    target = target.expanduser().resolve()
    archive = archive.expanduser().resolve()
    github_root = (ROOT / "GitHub").resolve()
    if ROOT.resolve() not in github_root.parents:
        raise ValueError("GitHub staging directory must remain within the source workspace")
    if target != github_root / "AF-CDMO":
        raise ValueError(f"Target must be {github_root / 'AF-CDMO'}")
    if archive.parent != github_root or archive.suffix.lower() != ".zip":
        raise ValueError(f"ZIP must be a direct child of {github_root}")
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Source and target must be separate directory trees")
    manifest = verify_release(source, strict=True)
    source_names = {entry["path"] for entry in manifest["files"]}

    old_names: set[str] = set()
    if target.exists():
        # Preserve an existing working tree, but refuse unrelated files before
        # overwriting anything. This also detects tampering with an old release.
        old_manifest = verify_release(target)
        old_names = {entry["path"] for entry in old_manifest["files"]}
        for path in target.rglob("*"):
            if path.is_symlink() and not is_runtime_metadata(path, target):
                raise RuntimeError(f"Refusing symbolic-link target: {path}")

    target.mkdir(parents=True, exist_ok=True)
    for relative in sorted(source_names):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
    for relative in sorted(old_names - source_names):
        destination = target / relative
        if destination.exists():
            destination.unlink()
    shutil.copy2(source / "release_manifest.json", target / "release_manifest.json")
    verify_release(target)

    archive.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{archive.name}.building-", suffix=".zip", dir=archive.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for relative in sorted(source_names | {"release_manifest.json"}):
                handle.write(source / relative, f"AF-CDMO/{relative}")
        temporary.replace(archive)
    finally:
        if temporary.exists():
            temporary.unlink()
    return len(source_names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    count = package_release(args.source, args.target, args.archive)
    print(f"[OK] GitHub staging folder: files={count} root={args.target.resolve()}")
    print(f"[OK] code-only upload archive: {args.archive.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
