"""Validate the unified deployment inference engine."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from deployment.inference.factory import InferenceFactory
from deployment.inference.monai_backend import MonaiBackend
from deployment.inference.nnunet_backend import NnUNetBackend
from deployment.inference.pipeline import UnifiedInferencePipeline
from deployment.registry import load_deployment_registries

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _pick_scan() -> Path:
    images = PROJECT_ROOT / "data" / "raw" / "label_192" / "images"
    candidates = sorted(images.glob("*.nii*"))
    if not candidates:
        raise FileNotFoundError(f"No BHSD images found under {images}")
    # Prefer a known thin volume used in prior validation when present.
    preferred = [
        images / "ID_0237f3c9_ID_40015688b9.nii.gz",
        images / "ID_0237f3c9_ID_40015688b9.nii",
    ]
    for path in preferred:
        if path.is_file():
            return path
    return candidates[0]


def _validate_result_schema(result, *, model_id: str, framework: str) -> None:
    _assert(result.model_id == model_id, f"model_id mismatch: {result.model_id} != {model_id}")
    _assert(result.framework == framework, f"framework mismatch: {result.framework}")
    _assert(result.prediction_mask.ndim == 3, "prediction_mask must be 3D")
    required_volume = {"total_volume_ml", "per_class_volume_ml", "classes_present", "spacing"}
    _assert(required_volume <= set(result.volumes), f"volumes missing keys: {required_volume - set(result.volumes)}")
    required_conf = {
        "study_confidence",
        "per_class_confidence",
        "mean_softmax_confidence",
        "confidence_histogram",
    }
    _assert(required_conf <= set(result.confidence), f"confidence missing keys: {required_conf - set(result.confidence)}")
    required_artifacts = {
        "prediction_mask",
        "overlay",
        "volume_report",
        "confidence",
        "prediction_summary",
        "prediction_report",
    }
    _assert(
        required_artifacts <= set(result.artifacts),
        f"artifacts missing: {required_artifacts - set(result.artifacts)}",
    )
    for key in required_artifacts:
        path = Path(result.artifacts[key])
        _assert(path.is_file(), f"Missing artifact file for '{key}': {path}")
    _assert(Path(result.report_path).is_file(), f"Missing report: {result.report_path}")


def run_validation(*, skip_nnunet: bool = False, project_root: Path | None = None) -> int:
    root = project_root or PROJECT_ROOT
    print("Unified inference engine validation")
    print(f"  project_root: {root}")

    _experiments, models = load_deployment_registries(project_root=root)
    factory = InferenceFactory(models, project_root=root)
    pipeline = UnifiedInferencePipeline(factory=factory)

    default_id = models.get_default().model_id
    default_backend = factory.create(None)
    _assert(isinstance(default_backend, MonaiBackend), "Default backend must be MonaiBackend")
    _assert(default_backend.model_id == default_id, "Default model_id mismatch")
    print(f"  default backend: {default_backend.model_id} ({default_backend.framework}) OK")

    nnunet_backend = factory.create("nnunet_fold0")
    _assert(isinstance(nnunet_backend, NnUNetBackend), "nnunet_fold0 must resolve to NnUNetBackend")
    print("  factory -> nnunet_fold0: NnUNetBackend OK")

    ssl_backend = factory.create("nnunet_ssl_2fold")
    _assert(isinstance(ssl_backend, NnUNetBackend), "nnunet_ssl_2fold must resolve to NnUNetBackend")
    _assert(ssl_backend._use_folds == (0, 1), "nnunet_ssl_2fold must use folds (0, 1)")
    print("  factory -> nnunet_ssl_2fold: NnUNetBackend folds=(0,1) OK")

    monai_backend = factory.create("monai_best30h")
    _assert(isinstance(monai_backend, MonaiBackend), "monai_best30h must resolve to MonaiBackend")
    print("  factory -> monai_best30h: MonaiBackend OK")

    # Structural check: backends implement the interface without duplicating MONAI pipeline class.
    import deployment.inference.monai_backend as monai_mod
    import deployment.inference.nnunet_backend as nnunet_mod

    monai_src = Path(monai_mod.__file__).read_text(encoding="utf-8")
    _assert("InferencePipeline" in monai_src, "MONAI backend must wrap InferencePipeline")
    _assert("class InferencePipeline" not in monai_src, "MONAI backend must not redefine InferencePipeline")
    nnunet_src = Path(nnunet_mod.__file__).read_text(encoding="utf-8")
    _assert("run_nnunet_prediction" in nnunet_src, "nnU-Net backend must reuse run_nnunet_prediction")
    _assert("prepare_nnunet_model_folder" in nnunet_src, "nnU-Net backend must reuse prepare_nnunet_model_folder")
    print("  no duplicated core prediction logic: OK")

    scan = _pick_scan()
    print(f"  test scan: {scan.name}")

    monai_out = root / "reports" / "inference" / "_validation" / "monai_best30h"
    monai_result = pipeline.run(scan, model_id="monai_best30h", output_dir=monai_out)
    _validate_result_schema(monai_result, model_id="monai_best30h", framework="monai")
    print(
        f"  MONAI prediction: OK "
        f"(volume={monai_result.volumes['total_volume_ml']:.4f} mL, "
        f"time={monai_result.metadata.get('processing_time_sec', 'n/a')})"
    )

    monai_fp = monai_result.schema_fingerprint()

    if skip_nnunet:
        print("  nnU-Net prediction: SKIPPED (--skip-nnunet)")
    else:
        if importlib.util.find_spec("nnunetv2") is None:
            print("VALIDATION FAILED")
            print("  - nnunetv2 is not installed; cannot validate nnU-Net backend")
            return 1
        nnunet_out = root / "reports" / "inference" / "_validation" / "nnunet_fold0"
        nnunet_result = pipeline.run(scan, model_id="nnunet_fold0", output_dir=nnunet_out)
        _validate_result_schema(nnunet_result, model_id="nnunet_fold0", framework="nnunet")
        nnunet_fp = nnunet_result.schema_fingerprint()
        _assert(
            monai_fp["keys"] == nnunet_fp["keys"],
            f"Result key schema mismatch: {monai_fp['keys']} vs {nnunet_fp['keys']}",
        )
        _assert(
            monai_fp["volume_keys"] == nnunet_fp["volume_keys"],
            "Volume schema mismatch between backends",
        )
        _assert(
            monai_fp["confidence_keys"] == nnunet_fp["confidence_keys"],
            "Confidence schema mismatch between backends",
        )
        print(
            f"  nnU-Net prediction: OK "
            f"(volume={nnunet_result.volumes['total_volume_ml']:.4f} mL, "
            f"time={nnunet_result.metadata.get('processing_time_sec', 'n/a')})"
        )
        print("  identical UnifiedInferenceResult schema: OK")

    print("VALIDATION PASSED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate unified inference engine.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--skip-nnunet",
        action="store_true",
        help="Skip the long nnU-Net prediction smoke (factory checks still run).",
    )
    args = parser.parse_args()
    try:
        code = run_validation(skip_nnunet=args.skip_nnunet, project_root=args.project_root)
    except Exception as exc:  # noqa: BLE001 — validation entrypoint reports failures
        print("VALIDATION FAILED")
        print(f"  - {exc}")
        raise SystemExit(1) from exc
    raise SystemExit(code)


if __name__ == "__main__":
    main()
