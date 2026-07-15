"""BHSD → nnU-Net v2 dataset conversion and locked-split helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "nnunet.yaml"

ALLOWED_LABELS = {0, 1, 2, 3, 4, 5}


def load_nnunet_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or DEFAULT_CONFIG
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dataset_folder_name(config: dict[str, Any]) -> str:
    return f"Dataset{int(config['dataset_id']):03d}_{config['dataset_name']}"


def case_id_from_filename(filename: str) -> str:
    name = Path(filename).name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return Path(name).stem


def resolve_volume_path(folder: Path, case_id: str) -> Path:
    """Resolve a case path for either ``.nii.gz`` or ``.nii`` on disk."""
    for suffix in (".nii.gz", ".nii"):
        candidate = folder / f"{case_id}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing volume for {case_id} under {folder}")


def load_split_groups(splits_csv: Path) -> dict[str, list[str]]:
    frame = pd.read_csv(splits_csv)
    if not {"filename", "split"}.issubset(frame.columns):
        raise ValueError(f"splits.csv must have filename,split columns: {splits_csv}")
    groups: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for _, row in frame.iterrows():
        split = str(row["split"]).strip().lower()
        if split not in groups:
            raise ValueError(f"Unexpected split value: {split}")
        groups[split].append(case_id_from_filename(str(row["filename"])))
    for key in groups:
        groups[key] = sorted(groups[key])
    return groups


def build_splits_final(groups: dict[str, list[str]]) -> list[dict[str, list[str]]]:
    """Single-fold nnU-Net split: train/val only. Test never appears."""
    overlap = set(groups["train"]) & set(groups["val"])
    if overlap:
        raise ValueError(f"train/val overlap: {sorted(overlap)[:5]}")
    test_leak = (set(groups["train"]) | set(groups["val"])) & set(groups["test"])
    if test_leak:
        raise ValueError(f"test leakage into train/val: {sorted(test_leak)[:5]}")
    return [{"train": list(groups["train"]), "val": list(groups["val"])}]


def write_splits_final_json(config: dict[str, Any], project_root: Path | None = None) -> Path:
    root = project_root or PROJECT_ROOT
    groups = load_split_groups(root / config["splits_csv"])
    payload = build_splits_final(groups)
    out_path = root / config["splits_final_json"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def _link_or_copy(src: Path, dst: Path, *, use_symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlink:
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _validate_pair(image_path: Path, label_path: Path) -> dict[str, Any]:
    image = nib.load(str(image_path))
    label = nib.load(str(label_path))
    image_data = np.asanyarray(image.dataobj)
    label_data = np.asanyarray(label.dataobj)
    if image_data.shape != label_data.shape:
        raise ValueError(
            f"Shape mismatch for {image_path.name}: image {image_data.shape} vs label {label_data.shape}"
        )
    unique = {int(v) for v in np.unique(label_data.astype(np.int16))}
    unexpected = sorted(unique - ALLOWED_LABELS)
    if unexpected:
        raise ValueError(f"Unexpected labels in {label_path.name}: {unexpected}")
    return {
        "shape": list(image_data.shape),
        "spacing": [float(x) for x in image.header.get_zooms()[:3]],
        "labels_present": sorted(unique),
    }


def convert_bhsd_to_nnunet(
    config: dict[str, Any],
    *,
    project_root: Path | None = None,
    use_symlink: bool = False,
    include_test_images: bool = True,
) -> dict[str, Any]:
    """Create nnU-Net raw dataset. Train+val in *Tr; test images optional in *Ts."""
    root = project_root or PROJECT_ROOT
    images_dir = root / config["images_dir"]
    labels_dir = root / config["labels_dir"]
    groups = load_split_groups(root / config["splits_csv"])

    dataset_dir = root / config["nnunet_raw_root"] / dataset_folder_name(config)
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    for folder in (images_tr, labels_tr, images_ts):
        folder.mkdir(parents=True, exist_ok=True)

    train_val_cases = groups["train"] + groups["val"]
    manifest_cases: list[dict[str, Any]] = []

    # Detect source ending from first train case; nnU-Net raw must use one consistent ending.
probe = resolve_volume_path(images_dir, train_val_cases[0])
source_ending = ".nii.gz" if probe.name.endswith(".nii.gz") else ".nii"
file_ending = source_ending  # keep as-on-disk; avoids recompressing 6GB+ on Kaggle

for case_id in train_val_cases:
        src_img = resolve_volume_path(images_dir, case_id)
        src_lab = resolve_volume_path(labels_dir, case_id)
        meta = _validate_pair(src_img, src_lab)
        dst_img = images_tr / f"{case_id}_0000{file_ending}"
        dst_lab = labels_tr / f"{case_id}{file_ending}"
        _link_or_copy(src_img, dst_img, use_symlink=use_symlink)
        _link_or_copy(src_lab, dst_lab, use_symlink=use_symlink)
        split_name = "train" if case_id in groups["train"] else "val"
        manifest_cases.append(
            {
                "case_id": case_id,
                "split": split_name,
                "source_ending": source_ending,
                **meta,
            }
        )

    test_cases_meta: list[dict[str, Any]] = []
    if include_test_images:
        for case_id in groups["test"]:
            src_img = resolve_volume_path(images_dir, case_id)
            src_lab = resolve_volume_path(labels_dir, case_id)
            meta = _validate_pair(src_img, src_lab)
            dst_img = images_ts / f"{case_id}_0000{file_ending}"
            _link_or_copy(src_img, dst_img, use_symlink=use_symlink)
            # Labels intentionally omitted from nnU-Net Ts; kept in BHSD for our eval.
            test_cases_meta.append(
                {
                    "case_id": case_id,
                    "split": "test",
                    "source_ending": source_ending,
                    **meta,
                }
            )

    dataset_json = {
        "channel_names": {str(k): v for k, v in dict(config["channel_names"]).items()},
        "labels": dict(config["labels"]),
        "numTraining": len(train_val_cases),
        "file_ending": file_ending,
    }
    (dataset_dir / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8"
    )

    splits_path = write_splits_final_json(config, project_root=root)

    manifest = {
        "dataset_folder": dataset_folder_name(config),
        "dataset_dir": str(dataset_dir),
        "num_train": len(groups["train"]),
        "num_val": len(groups["val"]),
        "num_test": len(groups["test"]),
        "numTraining_nnunet": len(train_val_cases),
        "splits_final_json": str(splits_path),
        "use_symlink": use_symlink,
        "cases_tr": manifest_cases,
        "cases_ts": test_cases_meta,
    }
    manifest_path = dataset_dir / "conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def validate_nnunet_raw(config: dict[str, Any], *, project_root: Path | None = None) -> dict[str, Any]:
    """Validate converted raw dataset + locked split integrity."""
    root = project_root or PROJECT_ROOT
    groups = load_split_groups(root / config["splits_csv"])
    dataset_dir = root / config["nnunet_raw_root"] / dataset_folder_name(config)
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    dataset_json_path = dataset_dir / "dataset.json"
    splits_path = root / config["splits_final_json"]

    errors: list[str] = []
    if not dataset_json_path.exists():
        errors.append(f"Missing {dataset_json_path}")
    if not splits_path.exists():
        errors.append(f"Missing {splits_path}")

    dataset_json = json.loads(dataset_json_path.read_text(encoding="utf-8")) if dataset_json_path.exists() else {}
    expected_tr = sorted(groups["train"] + groups["val"])
    label_files = []
    image_files = []
    if labels_tr.exists():
        label_files = sorted(
            p.name for p in labels_tr.iterdir() if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))
        )
    if images_tr.exists():
        image_files = sorted(
            p.name
            for p in images_tr.iterdir()
            if p.is_file() and ("_0000.nii" in p.name)
        )
    label_ids = sorted(case_id_from_filename(name) for name in label_files)
    image_ids = []
    for name in image_files:
        stem = case_id_from_filename(name)
        if stem.endswith("_0000"):
            stem = stem[: -len("_0000")]
        image_ids.append(stem)
    image_ids = sorted(image_ids)

    if label_ids != expected_tr:
        errors.append(
            f"labelsTr case set mismatch: got {len(label_ids)} expected {len(expected_tr)}"
        )
    if image_ids != expected_tr:
        errors.append(
            f"imagesTr case set mismatch: got {len(image_ids)} expected {len(expected_tr)}"
        )
    if int(dataset_json.get("numTraining", -1)) != len(expected_tr):
        errors.append(
            f"dataset.json numTraining={dataset_json.get('numTraining')} != {len(expected_tr)}"
        )

    # Spot-check every train/val pair for shape/label validity (full pass; local CPU OK).
    for case_id in expected_tr:
        img = None
        for suffix in (".nii.gz", ".nii"):
            candidate = images_tr / f"{case_id}_0000{suffix}"
            if candidate.exists():
                img = candidate
                break
        try:
            lab = resolve_volume_path(labels_tr, case_id)
        except FileNotFoundError:
            lab = None
        if img is None or lab is None:
            errors.append(f"Missing pair for {case_id}")
            continue
        try:
            _validate_pair(img, lab)
        except Exception as exc:  # noqa: BLE001 - collect all validation errors
            errors.append(f"{case_id}: {exc}")

    if splits_path.exists():
        splits = json.loads(splits_path.read_text(encoding="utf-8"))
        if not isinstance(splits, list) or len(splits) != 1:
            errors.append("splits_final.json must be a list with exactly 1 fold")
        else:
            fold0 = splits[0]
            if set(fold0.get("train", [])) != set(groups["train"]):
                errors.append("splits_final train list != splits.csv train")
            if set(fold0.get("val", [])) != set(groups["val"]):
                errors.append("splits_final val list != splits.csv val")
            if set(fold0.get("train", [])) & set(groups["test"]):
                errors.append("TEST LEAKAGE: test ids in splits_final train")
            if set(fold0.get("val", [])) & set(groups["test"]):
                errors.append("TEST LEAKAGE: test ids in splits_final val")

    # Ensure no test labels were placed in labelsTr
    test_in_tr = set(label_ids) & set(groups["test"])
    if test_in_tr:
        errors.append(f"TEST LEAKAGE: test cases in labelsTr: {sorted(test_in_tr)[:5]}")

    ok = not errors
    return {
        "ok": ok,
        "errors": errors,
        "num_train": len(groups["train"]),
        "num_val": len(groups["val"]),
        "num_test": len(groups["test"]),
        "numTraining": len(expected_tr),
        "dataset_dir": str(dataset_dir),
        "splits_final_json": str(splits_path),
    }
