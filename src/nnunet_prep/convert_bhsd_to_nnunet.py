"""CLI: convert BHSD label_192 into nnU-Net v2 raw dataset layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nnunet_prep import (
    convert_bhsd_to_nnunet,
    load_nnunet_config,
    validate_nnunet_raw,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config/nnunet.yaml",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Prefer symlinks instead of copying files (falls back to copy).",
    )
    parser.add_argument(
        "--skip-test-images",
        action="store_true",
        help="Do not populate imagesTs (test still held out of training).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate an existing conversion.",
    )
    args = parser.parse_args()
    config = load_nnunet_config(args.config)

    if args.validate_only:
        report = validate_nnunet_raw(config)
    else:
        manifest = convert_bhsd_to_nnunet(
            config,
            use_symlink=args.symlink,
            include_test_images=not args.skip_test_images,
        )
        print(json.dumps({k: manifest[k] for k in manifest if k not in {"cases_tr", "cases_ts"}}, indent=2))
        report = validate_nnunet_raw(config)

    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
