/** Types for API v1 predict — mirrors frozen contract. */

export type PredictResponse = {
  prediction_id: string;
  model_id: string;
  framework: string;
  category: string | null;
  scan_name: string;
  artifact_dir: string;
  volumes: {
    total_volume_ml?: number;
    per_class_volume_ml?: Record<string, number>;
    classes_present?: number[];
    spacing?: number[];
  };
  confidence: {
    study_confidence?: number;
    per_class_confidence?: Record<string, number>;
    mean_softmax_confidence?: number;
  };
  clinical: {
    severity_status: string;
    severity_tier: string | null;
    hemorrhage_detected: boolean;
    recommendations: string[];
    limitations: string[];
    disclaimer: string;
    summary: {
      subtypes?: string[];
      total_volume_ml?: number;
      study_confidence?: number | null;
      narrative?: string;
      hemorrhage_detected?: boolean;
    };
    clinical_report_path: string | null;
  };
  artifacts: Record<string, string>;
  download_urls: Record<string, string>;
  report_url: string;
  artifacts_url: string;
  metadata: {
    experiment_id?: string;
    device?: string;
    processing_time_sec?: number;
    cache_hit?: boolean;
    capabilities?: Record<string, boolean>;
  };
};
