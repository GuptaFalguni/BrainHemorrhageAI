"""
Precompute disk cache for all BHSD volumes used in training.

Run once before a long Kaggle/GPU job so epoch 1 does not re-preprocess
every slice access.

Example:
  python src/preprocessing/precache_volumes.py
  python src/preprocessing/precache_volumes.py --splits train val
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dataset.bhsd_dataset import VALID_SPLITS, load_split_filenames  # noqa: E402
from dataset.volume_cache import VolumeCache  # noqa: E402
from preprocessing.preprocess_volume import DEFAULT_CONFIG_PATH  # noqa: E402

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build BHSD preprocessed volume disk cache.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=sorted(VALID_SPLITS),
        help="Splits to cache (default: train val).",
    )
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to preprocessing.yaml",
    )
    args = parser.parse_args()

    _configure_logging()
    cache = VolumeCache(config_path=args.preprocessing_config)
    logger.info(
        "Disk cache dir: %s (fingerprint=%s, enabled=%s)",
        cache.disk_cache_dir,
        cache.fingerprint,
        cache.disk_cache_enabled,
    )

    started = time.perf_counter()
    grand = {"disk_hits": 0, "computed": 0, "total": 0}
    for split in args.splits:
        filenames = load_split_filenames(split)  # type: ignore[arg-type]
        logger.info("Caching split=%s files=%s", split, len(filenames))
        stats = cache.warm(filenames, config_path=args.preprocessing_config)
        for key in grand:
            grand[key] += stats[key]
        logger.info("split=%s stats=%s", split, stats)

    elapsed = time.perf_counter() - started
    logger.info("Done in %.1fs — %s", elapsed, grand)
    logger.info("Cache files now: %s", len(list(cache.disk_cache_dir.glob("*.npz"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
