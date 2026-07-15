"""Validate deployment experiment and model registries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from deployment.registry import load_deployment_registries

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_validation(*, project_root: Path | None = None) -> int:
    root = project_root or PROJECT_ROOT
    print("Deployment registry validation")
    print(f"  project_root: {root}")

    experiments, models = load_deployment_registries(project_root=root)
    errors = models.validate(require_checkpoints=True)

    print(f"  experiments: {len(experiments.records)}")
    for experiment_id in experiments.list_ids():
        record = experiments.get(experiment_id)
        print(
            f"    - {experiment_id}: checkpoint={'OK' if record.checkpoint_exists else 'MISSING'} "
            f"epochs={record.epochs} status={record.publication_status}"
        )

    print(f"  models: {len(models.records)}")
    for model_id in models.list_ids():
        model = models.get(model_id)
        print(
            f"    - {model_id}: experiment={model.experiment_id} "
            f"category={model.category} default={model.default}"
        )

    default = models.get_default()
    print(f"  default_model: {default.model_id}")

    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("VALIDATION PASSED")
    print("  schema: OK")
    print("  checkpoints: OK")
    print("  experiment references: OK")
    print("  exactly one default model: OK")
    print("  capabilities: OK")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate deployment registries.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Optional project root override.",
    )
    args = parser.parse_args()
    raise SystemExit(run_validation(project_root=args.project_root))


if __name__ == "__main__":
    main()
