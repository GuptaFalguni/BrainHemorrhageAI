/** Types mirroring API v1 health contract. */

export type HealthResponse = {
  status: string;
  version: string;
  available_models: string[];
  default_model_id: string | null;
  device: string;
  gpu_available: boolean;
  python_version: string;
  pytorch_version: string;
  monai_version: string | null;
  registry_loaded: boolean;
};
