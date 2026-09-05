#!/bin/sh
set -eu

APP_ROOT="${APP_ROOT:-/app}"
CHECKPOINT_ROOT="${APP_ROOT}/checkpoints"
WEIGHTS_URL="${WEIGHTS_URL:-https://github.com/GuptaFalguni/BrainHemorrhageAI/releases/download/v1.1.0/model-weights-v1.1.0.tar.gz}"

MONAI_CKPT="${CHECKPOINT_ROOT}/experiment_best30h/best_model.pt"
NNUNET_CKPT="${CHECKPOINT_ROOT}/nnunet_dataset501_fold0/checkpoint_best.pth"
NNUNET_PLANS="${CHECKPOINT_ROOT}/nnunet_dataset501_fold0/nnUNetPlans.json"
NNUNET_DATASET="${CHECKPOINT_ROOT}/nnunet_dataset501_fold0/dataset.json"
SSL_F0="${CHECKPOINT_ROOT}/nnunet_ssl_2fold/fold_0/checkpoint_best.pth"
SSL_F1="${CHECKPOINT_ROOT}/nnunet_ssl_2fold/fold_1/checkpoint_best.pth"
SSL_PLANS="${CHECKPOINT_ROOT}/nnunet_ssl_2fold/nnUNetPlans.json"
SSL_DATASET="${CHECKPOINT_ROOT}/nnunet_ssl_2fold/dataset.json"

weights_ready() {
  [ -f "${MONAI_CKPT}" ] \
    && [ -f "${NNUNET_CKPT}" ] \
    && [ -f "${NNUNET_PLANS}" ] \
    && [ -f "${NNUNET_DATASET}" ] \
    && [ -f "${SSL_F0}" ] \
    && [ -f "${SSL_F1}" ] \
    && [ -f "${SSL_PLANS}" ] \
    && [ -f "${SSL_DATASET}" ]
}

if weights_ready; then
  echo "[entrypoint] Model weights already present — skipping download."
else
  echo "[entrypoint] Model weights missing — downloading from:"
  echo "  ${WEIGHTS_URL}"
  mkdir -p "${CHECKPOINT_ROOT}"
  tmp_archive="$(mktemp /tmp/model-weights.XXXXXX.tar.gz)"
  if ! curl -fL --retry 3 --retry-delay 2 -o "${tmp_archive}" "${WEIGHTS_URL}"; then
    echo "[entrypoint] ERROR: Failed to download weights."
    echo "  Publish release asset or set WEIGHTS_URL / mount ./checkpoints."
    rm -f "${tmp_archive}"
    exit 1
  fi
  tar -xzf "${tmp_archive}" -C "${APP_ROOT}"
  rm -f "${tmp_archive}"
  if ! weights_ready; then
    echo "[entrypoint] ERROR: Archive extracted but required checkpoint files are still missing."
    exit 1
  fi
  echo "[entrypoint] Weights installed."
fi

mkdir -p "${APP_ROOT}/reports/api_v1/predictions"
exec "$@"
