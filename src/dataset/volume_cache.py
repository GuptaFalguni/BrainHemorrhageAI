"""
Memory + disk cache for preprocessed BHSD volumes.

Why:
  Slice-based 2.5D training touches the same volume many times per epoch.
  A tiny RAM LRU (old default: 4) plus DataLoader workers caused repeated
  NIfTI preprocess calls and multi-hour stalls on Kaggle before epoch 1.

Approach:
  - Optional on-disk ``.npz`` cache keyed by filename + preprocessing fingerprint
  - Small per-process RAM LRU for slice locality within a worker
  - Atomic writes so multi-worker first-touch is safe
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from preprocessing.preprocess_volume import (
    PreprocessedVolume,
    load_preprocessing_config,
    preprocess_volume,
    resolve_clip_bounds,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CACHE_FORMAT_VERSION = 1


def _config_fingerprint(config: dict[str, Any]) -> str:
    clip_min, clip_max = resolve_clip_bounds(config)
    payload = {
        "format": _CACHE_FORMAT_VERSION,
        "clip_min": clip_min,
        "clip_max": clip_max,
        "normalization_method": str(config.get("normalization_method", "none")).lower(),
        "train_mean": config.get("train_mean"),
        "train_std": config.get("train_std"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _safe_stem(filename: str) -> str:
    name = Path(filename).name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return Path(name).stem


class VolumeCache:
    """LRU RAM cache with optional persistent disk backend."""

    def __init__(
        self,
        max_entries: int | None = None,
        disk_cache_dir: Path | str | None = None,
        disk_cache_enabled: bool | None = None,
        config_path: Path | None = None,
    ) -> None:
        self._config_path = config_path
        config = load_preprocessing_config(config_path)

        if max_entries is None:
            max_entries = int(config.get("memory_cache_entries", 16))
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1.")

        if disk_cache_enabled is None:
            disk_cache_enabled = bool(config.get("disk_cache_enabled", True))

        if disk_cache_dir is None:
            disk_cache_dir = config.get("disk_cache_dir", "data/processed/volume_cache")
        disk_path = Path(disk_cache_dir)
        if not disk_path.is_absolute():
            disk_path = PROJECT_ROOT / disk_path

        self._max_entries = max_entries
        self._disk_cache_enabled = disk_cache_enabled
        self._disk_cache_dir = disk_path
        self._fingerprint = _config_fingerprint(config)
        self._store: dict[str, PreprocessedVolume] = {}
        self._order: list[str] = []

        if self._disk_cache_enabled:
            self._disk_cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def disk_cache_dir(self) -> Path:
        return self._disk_cache_dir

    @property
    def disk_cache_enabled(self) -> bool:
        return self._disk_cache_enabled

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def cache_path_for(self, filename: str) -> Path:
        return self._disk_cache_dir / f"{_safe_stem(filename)}__{self._fingerprint}.npz"

    def get(self, filename: str, config_path: Path | None = None) -> PreprocessedVolume:
        """Return a preprocessed volume from RAM, disk, or fresh compute."""
        resolved_config = config_path or self._config_path
        if filename in self._store:
            self._touch(filename)
            return self._store[filename]

        volume: PreprocessedVolume | None = None
        if self._disk_cache_enabled:
            volume = self._load_disk(filename)

        if volume is None:
            volume = preprocess_volume(filename, config_path=resolved_config)
            if self._disk_cache_enabled:
                self._save_disk(volume)

        self._put_ram(filename, volume)
        return volume

    def warm(
        self,
        filenames: list[str],
        config_path: Path | None = None,
    ) -> dict[str, int]:
        """Ensure all filenames exist on disk cache. Returns hit/miss counts."""
        resolved_config = config_path or self._config_path
        hits = 0
        misses = 0
        for filename in filenames:
            path = self.cache_path_for(filename)
            if self._disk_cache_enabled and path.exists():
                hits += 1
                continue
            volume = preprocess_volume(filename, config_path=resolved_config)
            if self._disk_cache_enabled:
                self._save_disk(volume)
            misses += 1
        return {"disk_hits": hits, "computed": misses, "total": len(filenames)}

    def clear(self) -> None:
        """Drop the in-memory LRU only (disk cache retained)."""
        self._store.clear()
        self._order.clear()

    def _touch(self, filename: str) -> None:
        if filename in self._order:
            self._order.remove(filename)
        self._order.append(filename)

    def _put_ram(self, filename: str, volume: PreprocessedVolume) -> None:
        self._store[filename] = volume
        self._touch(filename)
        while len(self._order) > self._max_entries:
            evicted = self._order.pop(0)
            self._store.pop(evicted, None)

    def _load_disk(self, filename: str) -> PreprocessedVolume | None:
        path = self.cache_path_for(filename)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                if int(data["format_version"]) != _CACHE_FORMAT_VERSION:
                    return None
                stored_fp = data["fingerprint"].tobytes().decode("utf-8")
                if stored_fp != self._fingerprint:
                    return None
                stored_name = data["filename"].tobytes().decode("utf-8")
                if stored_name != filename and _safe_stem(stored_name) != _safe_stem(filename):
                    return None
                volume = PreprocessedVolume(
                    processed_image=np.array(data["processed_image"], dtype=np.float32),
                    processed_mask=np.array(data["processed_mask"], dtype=np.int64),
                    original_spacing=tuple(float(v) for v in data["original_spacing"].tolist()),
                    original_shape=tuple(int(v) for v in data["original_shape"].tolist()),
                    affine=np.array(data["affine"], dtype=np.float64),
                    filename=filename,
                )
            logger.debug("Disk cache hit: %s", filename)
            return volume
        except Exception as exc:  # noqa: BLE001 — corrupt cache should recompute
            logger.warning("Disk cache unreadable for %s (%s); recomputing.", filename, exc)
            return None

    def _save_disk(self, volume: PreprocessedVolume) -> None:
        path = self.cache_path_for(volume.filename)
        self._disk_cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.stem}.tmp.npz")
        try:
            np.savez(
                tmp_path,
                format_version=np.int32(_CACHE_FORMAT_VERSION),
                fingerprint=np.frombuffer(self._fingerprint.encode("utf-8"), dtype=np.uint8),
                filename=np.frombuffer(volume.filename.encode("utf-8"), dtype=np.uint8),
                processed_image=np.asarray(volume.processed_image, dtype=np.float32),
                processed_mask=np.asarray(volume.processed_mask, dtype=np.int64),
                original_spacing=np.asarray(volume.original_spacing, dtype=np.float64),
                original_shape=np.asarray(volume.original_shape, dtype=np.int64),
                affine=np.asarray(volume.affine, dtype=np.float64),
            )
            os.replace(tmp_path, path)
            logger.debug("Disk cache write: %s", path.name)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise
