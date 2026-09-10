export const API_MODEL_IDS = {
  monai: "monai_best30h",
  nnunet: "nnunet_fold0",
  ssl: "nnunet_ssl_2fold",
  semi: "nnunet_dataset502_semi",
} as const;

export type ComparePresetId =
  | "all"
  | "monai-nnunet"
  | "monai-ssl"
  | "nnunet-ssl"
  | "ssl-semi"
  | "nnunet-semi";

export const COMPARE_PRESETS: {
  id: ComparePresetId;
  label: string;
  modelIds: string[];
}[] = [
  {
    id: "all",
    label: "All 4 models — MONAI, nnU-Net, SSL, Semi-sup.",
    modelIds: [
      API_MODEL_IDS.monai,
      API_MODEL_IDS.nnunet,
      API_MODEL_IDS.ssl,
      API_MODEL_IDS.semi,
    ],
  },
  {
    id: "monai-nnunet",
    label: "MONAI + nnU-Net",
    modelIds: [API_MODEL_IDS.monai, API_MODEL_IDS.nnunet],
  },
  {
    id: "monai-ssl",
    label: "MONAI + SSL nnU-Net",
    modelIds: [API_MODEL_IDS.monai, API_MODEL_IDS.ssl],
  },
  {
    id: "nnunet-ssl",
    label: "nnU-Net + SSL nnU-Net",
    modelIds: [API_MODEL_IDS.nnunet, API_MODEL_IDS.ssl],
  },
  {
    id: "ssl-semi",
    label: "SSL nnU-Net + Semi-sup. nnU-Net",
    modelIds: [API_MODEL_IDS.ssl, API_MODEL_IDS.semi],
  },
  {
    id: "nnunet-semi",
    label: "nnU-Net + Semi-sup. nnU-Net",
    modelIds: [API_MODEL_IDS.nnunet, API_MODEL_IDS.semi],
  },
];

export function modelIdsForPreset(preset: ComparePresetId): string[] {
  return (
    COMPARE_PRESETS.find((p) => p.id === preset)?.modelIds ??
    COMPARE_PRESETS[0].modelIds
  );
}
