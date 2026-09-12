#!/bin/sh
set -eu

APP_ROOT="${APP_ROOT:-/app}"
CHECKPOINT_ROOT="${APP_ROOT}/checkpoints"
WEIGHTS_URL="${WEIGHTS_URL:-https://github.com/GuptaFalguni/BrainHemorrhageAI/releases/download/v1.1.0/model-weights-v1.1.0.tar.gz}"
SSL_WEIGHTS_URL="${SSL_WEIGHTS_URL:-https://github.com/GuptaFalguni/BrainHemorrhageAI/releases/download/v1.2.0/model-weights-ssl-dataset502.tar.gz}"

MONAI_CKPT="${CHECKPOINT_ROOT}/experiment_best30h/best_model.pt"
NNUNET_CKPT="${CHECKPOINT_ROOT}/nnunet_dataset501_fold0/checkpoint_best.pth"
NNUNET_PLANS="${CHECKPOINT_ROOT}/nnunet_dataset501_fold0/nnUNetPlans.json"
NNUNET_DATASET="${CHECKPOINT_ROOT}/nnunet_dataset501_fold0/dataset.json"
SSL_F0="${CHECKPOINT_ROOT}/nnunet_ssl_2fold/fold_0/checkpoint_best.pth"
SSL_F1="${CHECKPOINT_ROOT}/nnunet_ssl_2fold/fold_1/checkpoint_best.pth"
SSL_PLANS="${CHECKPOINT_ROOT}/nnunet_ssl_2fold/nnUNetPlans.json"
SSL_DATASET="${CHECKPOINT_ROOT}/nnunet_ssl_2fold/dataset.json"
SEMI_DIR="${CHECKPOINT_ROOT}/nnunet_dataset502_semi"
SEMI_CKPT="${SEMI_DIR}/checkpoint_best.pth"
SEMI_SIDECARS="${APP_ROOT}/config/models/nnunet_dataset502_semi"

core_ready() {
  [ -f "${MONAI_CKPT}" ] \
    && [ -f "${NNUNET_CKPT}" ] \
    && [ -f "${NNUNET_PLANS}" ] \
    && [ -f "${NNUNET_DATASET}" ] \
    && [ -f "${SSL_F0}" ] \
    && [ -f "${SSL_F1}" ] \
    && [ -f "${SSL_PLANS}" ] \
    && [ -f "${SSL_DATASET}" ]
}

ssl_ready() {
  [ -f "${SEMI_CKPT}" ]
}

copy_semi_sidecars() {
  mkdir -p "${SEMI_DIR}"
  if [ -d "${SEMI_SIDECARS}" ]; then
    for name in dataset.json nnUNetPlans.json plans.json; do
      if [ -f "${SEMI_SIDECARS}/${name}" ] && [ ! -f "${SEMI_DIR}/${name}" ]; then
        cp "${SEMI_SIDECARS}/${name}" "${SEMI_DIR}/${name}"
      fi
    done
  fi
}

download_archive() {
  url="$1"
  label="$2"
  echo "[entrypoint] ${label} missing — downloading from:"
  echo "  ${url}"
  mkdir -p "${CHECKPOINT_ROOT}"
  tmp_archive="$(mktemp /tmp/model-weights.XXXXXX.tar.gz)"
  if ! curl -fL --retry 3 --retry-delay 2 -o "${tmp_archive}" "${url}"; then
    echo "[entrypoint] ERROR: Failed to download ${label}."
    echo "  Publish release asset or set WEIGHTS_URL / SSL_WEIGHTS_URL / mount ./checkpoints."
    rm -f "${tmp_archive}"
    exit 1
  fi
  tar -xzf "${tmp_archive}" -C "${APP_ROOT}"
  rm -f "${tmp_archive}"
}

if core_ready; then
  echo "[entrypoint] MONAI + nnU-Net + Ensemble weights already present — skipping core download."
else
  download_archive "${WEIGHTS_URL}" "MONAI / nnU-Net / Ensemble weights"
  if ! core_ready; then
    echo "[entrypoint] ERROR: Core archive extracted but required checkpoint files are still missing."
    exit 1
  fi
  echo "[entrypoint] Core weights installed."
fi

copy_semi_sidecars

if ssl_ready; then
  echo "[entrypoint] SSL (semi-supervised) weights already present."
else
  download_archive "${SSL_WEIGHTS_URL}" "SSL (Dataset502) weights"
  copy_semi_sidecars
  if ! ssl_ready; then
    echo "[entrypoint] ERROR: SSL archive extracted but checkpoints/nnunet_dataset502_semi/checkpoint_best.pth is still missing."
    exit 1
  fi
  echo "[entrypoint] SSL weights installed."
fi

mkdir -p "${APP_ROOT}/reports/api_v1/predictions"
exec "$@"
