import type { PredictResponse } from "@/types/predict";
import type { ModelResult, SubtypeVolume } from "@/stores/workspace-store";

const LABEL_TO_CODE: Record<string, string> = {
  "1": "EDH",
  "2": "SDH",
  "3": "SAH",
  "4": "IPH",
  "5": "IVH",
  EDH: "EDH",
  SDH: "SDH",
  SAH: "SAH",
  IPH: "IPH",
  IVH: "IVH",
};

const SUBTYPE_FULL: Record<string, string> = {
  EDH: "Epidural Hemorrhage",
  SDH: "Subdural Hemorrhage",
  SAH: "Subarachnoid Hemorrhage",
  IPH: "Intraparenchymal Hemorrhage",
  IVH: "Intraventricular Hemorrhage",
};

const COLOR_HINT: Record<string, string> = {
  EDH: "Red",
  SDH: "Yellow",
  SAH: "Cyan",
  IPH: "Green",
  IVH: "Magenta",
};

export { COLOR_HINT };

function buildSubtypeVolumes(response: PredictResponse): SubtypeVolume[] {
  const perClass = response.volumes.per_class_volume_ml ?? {};
  const entries = Object.entries(perClass)
    .map(([key, ml]) => ({
      code: LABEL_TO_CODE[key] ?? key,
      volumeMl: Number(ml) || 0,
    }))
    .filter((e) => e.volumeMl > 0)
    .sort((a, b) => b.volumeMl - a.volumeMl);

  const total =
    Number(response.volumes.total_volume_ml) ||
    entries.reduce((s, e) => s + e.volumeMl, 0) ||
    1;

  return entries.map((e, index) => ({
    code: e.code,
    fullName: SUBTYPE_FULL[e.code] ?? e.code,
    volumeMl: e.volumeMl,
    percent: (e.volumeMl / total) * 100,
    isPrimary: index === 0,
  }));
}

export function mapPredictToResult(response: PredictResponse): ModelResult {
  const subtypeVolumes = buildSubtypeVolumes(response);
  const primary = subtypeVolumes[0];
  const code = primary?.code ?? (response.clinical.hemorrhage_detected ? "Detected" : "None");
  const subtypeFull =
    primary?.fullName ??
    (response.clinical.hemorrhage_detected
      ? "Hemorrhage detected"
      : "No hemorrhage detected");

  const volumeMl = Number(
    response.volumes.total_volume_ml ??
      response.clinical.summary?.total_volume_ml ??
      subtypeVolumes.reduce((s, e) => s + e.volumeMl, 0),
  );
  const confidence = Number(
    response.confidence.study_confidence ??
      response.clinical.summary?.study_confidence ??
      0,
  );
  const processingSeconds = Number(response.metadata.processing_time_sec ?? 0);

  const displayName =
    response.model_id.includes("nnunet") ? "nnU-Net" : "MONAI";

  return {
    modelId: response.model_id,
    displayName,
    predictionId: response.prediction_id,
    subtype: code,
    subtypeFull,
    volumeMl,
    confidence,
    inferenceSeconds: processingSeconds,
    processingSeconds,
    hemorrhageDetected: response.clinical.hemorrhage_detected,
    subtypeVolumes,
    overlaySrc: "",
    overlay3dSrc: null,
  };
}

export function modelIdForMode(mode: "monai" | "nnunet" | "compare"): string[] {
  if (mode === "monai") return ["monai_best30h"];
  if (mode === "nnunet") return ["nnunet_fold0"];
  return ["monai_best30h", "nnunet_fold0"];
}
