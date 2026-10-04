"""Fail-closed checks for the code-only public distribution boundary."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts import build_public_release as builder
from scripts import package_public_release as packaging
from scripts.verify_release_manifest import verify_release


def _release(root: Path, *, text: str = "example") -> dict:
    root.mkdir(parents=True, exist_ok=True)
    payload = text.encode("utf-8")
    (root / "README.md").write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "release_type": "code_only",
        "raw_market_data_included": False,
        "trained_weights_included": False,
        "timestamp_level_derivatives_included": False,
        "jao_authorization_included": False,
        "files": [{"path": "README.md", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}],
    }
    _write_manifest(root, manifest)
    return manifest


def _write_manifest(root: Path, manifest: dict) -> None:
    (root / "release_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_manifest_detects_content_tampering(tmp_path: Path) -> None:
    _release(tmp_path)
    verify_release(tmp_path, strict=True)
    (tmp_path / "README.md").write_text("modified", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash_or_size_mismatch"):
        verify_release(tmp_path)


@pytest.mark.parametrize("relative", ["build/records.npz", "dist/weights.pt", "__pycache__/records.npz", "unexpected.pyc"])
def test_manifest_rejects_hidden_unexpected_outputs(tmp_path: Path, relative: str) -> None:
    _release(tmp_path)
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"local output")
    with pytest.raises(RuntimeError, match="unexpected"):
        verify_release(tmp_path)


def test_runtime_caches_are_accepted_only_outside_strict_verification(tmp_path: Path) -> None:
    _release(tmp_path)
    cache = tmp_path / "__pycache__" / "example.cpython-310.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"bytecode")
    verify_release(tmp_path)
    with pytest.raises(RuntimeError, match="unexpected"):
        verify_release(tmp_path, strict=True)


@pytest.mark.parametrize("relative", ["../README.md", "/README.md", "C:/README.md", "scripts\\example.py", "records.npz"])
def test_manifest_rejects_unsafe_or_restricted_entries(tmp_path: Path, relative: str) -> None:
    manifest = _release(tmp_path)
    manifest["files"][0]["path"] = relative
    _write_manifest(tmp_path, manifest)
    with pytest.raises(RuntimeError, match="Unsafe or non-code"):
        verify_release(tmp_path)


def test_manifest_rejects_duplicate_paths_and_false_boundary_flags(tmp_path: Path) -> None:
    manifest = _release(tmp_path)
    manifest["files"].append(dict(manifest["files"][0]))
    _write_manifest(tmp_path, manifest)
    with pytest.raises(RuntimeError, match="duplicate"):
        verify_release(tmp_path)
    manifest["files"].pop()
    manifest["raw_market_data_included"] = True
    _write_manifest(tmp_path, manifest)
    with pytest.raises(RuntimeError, match="metadata"):
        verify_release(tmp_path)


def test_failed_rebuild_preserves_previous_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    output = tmp_path / "public_release" / "example"
    _release(output)

    def fail(staging: Path) -> dict:
        (staging / "incomplete.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("missing required source")

    monkeypatch.setattr(builder, "populate_release", fail)
    with pytest.raises(RuntimeError, match="missing required source"):
        builder.build_release(output, force=True)
    verify_release(output, strict=True)
    assert not list(output.parent.glob(".example.building-*"))


def test_builder_supports_a_public_checkout_and_tracks_the_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "ROOT_FILES", {"README_PUBLIC.md": "README.md"})
    for registry in ("SCRIPT_FILES", "TEST_FILES", "QCT_FILES", "LEARNING_HEADROOM_FILES"):
        monkeypatch.setattr(builder, registry, ())
    (tmp_path / "README.md").write_text("public README", encoding="utf-8")
    output = tmp_path / "public_release" / "example"
    builder.build_release(output)
    verify_release(output, strict=True)
    assert (output / "README.md").read_text(encoding="utf-8") == "public README"
    ignores = (output / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "!release_manifest.json" in ignores
    assert ".venv/" in ignores


def test_package_preserves_git_and_omits_runtime_caches_from_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(packaging, "ROOT", tmp_path)
    source = tmp_path / "public_release" / "example"
    target = tmp_path / "GitHub" / "AF-CDMO"
    archive = tmp_path / "GitHub" / "AF-CDMO-code-only.zip"
    _release(source, text="updated")
    _release(target, text="old")
    git = target / ".git"
    git.mkdir()
    (git / "config").write_text("private Git config", encoding="utf-8")
    cache = target / "__pycache__"
    cache.mkdir()
    (cache / "example.pyc").write_bytes(b"bytecode")
    packaging.package_release(source, target, archive)
    assert (git / "config").read_text(encoding="utf-8") == "private Git config"
    assert (target / "README.md").read_text(encoding="utf-8") == "updated"
    with zipfile.ZipFile(archive) as handle:
        assert sorted(handle.namelist()) == ["AF-CDMO/README.md", "AF-CDMO/release_manifest.json"]


def test_package_refuses_unmanaged_target_files_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(packaging, "ROOT", tmp_path)
    source = tmp_path / "public_release" / "example"
    target = tmp_path / "GitHub" / "AF-CDMO"
    archive = tmp_path / "GitHub" / "AF-CDMO-code-only.zip"
    _release(source, text="updated")
    _release(target, text="old")
    (target / "private.npz").write_bytes(b"restricted")
    with pytest.raises(RuntimeError, match="unexpected"):
        packaging.package_release(source, target, archive)
    assert (target / "README.md").read_text(encoding="utf-8") == "old"
