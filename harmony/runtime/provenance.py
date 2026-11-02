"""Capture local code and resource identities without loading models or calling APIs."""

from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

from harmony import __version__


_CACHE_REPOSITORIES = ("facebook/sam-audio-large", "stabilityai/stable-audio-3-medium")
_GENERATED_DIRECTORIES = {"__pycache__", "outputs", "generated", "evidence"}
_COMMIT_ID = re.compile(r"[0-9a-fA-F]{40}")


def _git_identity(path: Path) -> dict:
    """Only attribute a checkout when its Git root is the requested directory."""
    result = {"status": "unavailable", "commit": None, "dirty": None}
    if not path.is_dir():
        result["reason"] = "directory_missing"
        return result

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        ).stdout.strip()

    try:
        actual_root = Path(git("rev-parse", "--show-toplevel")).resolve()
        if actual_root != path.resolve():
            result["reason"] = "directory_is_not_repository_root"
            return result
        result["commit"] = git("rev-parse", "HEAD")
        result["status"] = "available"
        try:
            result["dirty"] = bool(git("status", "--porcelain", "--untracked-files=normal"))
        except (OSError, subprocess.SubprocessError):
            result["dirty_reason"] = "git_status_unavailable"
    except (OSError, subprocess.SubprocessError):
        result["reason"] = "git_identity_unavailable"
    return result


def _source_identity(source: Path) -> dict:
    """Hash Python and JSON source files; exclude generated audio/evidence folders."""
    files = []
    unreadable = []
    missing_packages = []
    for package in ("audio_edit", "harmony"):
        folder = source / package
        if not folder.is_dir():
            missing_packages.append(package)
            continue
        for current, directories, names in os.walk(folder, followlinks=False):
            directories[:] = sorted(
                name
                for name in directories
                if name not in _GENERATED_DIRECTORIES
                and not name.startswith(".")
                and not (Path(current) / name).is_symlink()
            )
            for name in sorted(names):
                path = Path(current) / name
                if path.suffix not in {".py", ".json"}:
                    continue
                relative = path.relative_to(source).as_posix()
                if path.is_symlink():
                    unreadable.append(relative)
                    continue
                try:
                    digest = sha256()
                    with path.open("rb") as handle:
                        for block in iter(lambda: handle.read(65536), b""):
                            digest.update(block)
                    files.append({"path": relative, "sha256": digest.hexdigest()})
                except OSError:
                    unreadable.append(relative)
    files.sort(key=lambda item: item["path"])
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    complete = bool(files) and not unreadable and not missing_packages
    return {
        "status": "available" if complete else "incomplete",
        "algorithm": "sha256",
        "aggregate_encoding": "UTF-8 canonical JSON of sorted path/sha256 entries",
        "sha256": sha256(encoded).hexdigest() if complete else None,
        "file_count": len(files),
        "files": files,
        "unreadable_files": unreadable,
        "missing_packages": missing_packages,
    }


def _hub_cache() -> Path:
    configured = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if configured:
        return Path(configured).expanduser()
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home).expanduser() / "hub"
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache").expanduser()
    return cache / "huggingface" / "hub"


def _cached_model_identity(cache: Path, repository: str) -> dict:
    """Read small local refs and snapshot directory names, never model weights."""
    folder = cache / ("models--" + repository.replace("/", "--"))
    result = {
        "repository": repository,
        "status": "not_cached",
        "refs": {},
        "snapshot_ids": [],
        "used_revision": None,
    }
    if not folder.is_dir():
        return result
    result["status"] = "cached"
    refs = folder / "refs"
    try:
        if refs.is_dir():
            count = 0
            for current, directories, names in os.walk(refs, followlinks=False):
                directories[:] = [
                    name for name in directories if not (Path(current) / name).is_symlink()
                ]
                if len(Path(current).relative_to(refs).parts) >= 2:
                    directories[:] = []
                for name in names:
                    count += 1
                    if count > 32:
                        result["refs_truncated"] = True
                        break
                    path = Path(current) / name
                    if path.is_symlink():
                        continue
                    with path.open("r", encoding="utf-8") as handle:
                        revision = handle.read(4096).strip()
                    if _COMMIT_ID.fullmatch(revision):
                        result["refs"][path.relative_to(refs).as_posix()] = revision
                if count > 32:
                    break
        snapshots = folder / "snapshots"
        if snapshots.is_dir():
            with os.scandir(snapshots) as entries:
                for index, entry in enumerate(entries):
                    if index >= 128:
                        result["snapshots_truncated"] = True
                        break
                    if entry.is_dir(follow_symlinks=False) and _COMMIT_ID.fullmatch(entry.name):
                        result["snapshot_ids"].append(entry.name)
        result["snapshot_ids"].sort()
        result["refs"] = dict(sorted(result["refs"].items()))
    except (OSError, UnicodeError):
        result["status"] = "partially_readable"
    return result


def capture(root: Path, resource: Path, source: Path | None = None) -> dict:
    """Record local provenance for the launcher and its copied source tree.

    Cached model revisions describe availability, not which revision a remote
    service or later model load actually uses. No credentials or environment
    dump are included, and missing Git/cache information remains explicitly unknown.
    """
    root, resource = Path(root).resolve(), Path(resource).resolve()
    snapshot = Path(source).resolve() if source is not None else root
    cache = _hub_cache()
    return {
        "harmony_version": __version__,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "repository": _git_identity(root),
        "source": _source_identity(snapshot),
        "external_repositories": {
            name: _git_identity(resource / "externals" / name)
            for name in ("sam-audio", "stable-audio-3")
        },
        "huggingface_cache": {
            "scope": "local cache inventory only; not proof of the loaded model revision",
            "models": [_cached_model_identity(cache, repo) for repo in _CACHE_REPOSITORIES],
        },
    }
