export interface Detection {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
  score: number;
  label: string;
  center_x: number;
  center_y: number;
}

export interface Prediction {
  model: string;
  detections: Detection[];
  detection_count: number;
  image_score: number;
  has_foreign_object: boolean;
  conf_threshold: number;
  image_width: number;
  image_height: number;
  inference_ms: number;
  source: "model" | "demo";
  filename: string | null;
}

export interface CompareResult {
  filename: string | null;
  image_width: number;
  image_height: number;
  results: Prediction[];
  agreement: string;
}

export interface BatchItem {
  filename: string;
  ok: boolean;
  error: string | null;
  prediction: Prediction | null;
}

export interface BatchResult {
  items: BatchItem[];
  total: number;
  positive: number;
  negative: number;
  failed: number;
  total_ms: number;
}

export interface ModelInfo {
  name: string;
  display_name: string;
  description: string;
  input_size: number;
  default_conf: number;
  weights_path: string;
  weights_present: boolean;
  dependencies_available: boolean;
  ready: boolean;
  loaded: boolean;
  has_metrics: boolean;
}

export interface ModelList {
  models: ModelInfo[];
  device: string;
  demo_mode: boolean;
  any_ready: boolean;
}

export interface Health {
  status: string;
  app_name: string;
  version: string;
  environment: string;
  device: string;
  torch_version: string;
  torchvision_version: string;
  ultralytics_available: boolean;
  models_ready: string[];
  dataset_available: boolean;
}

export interface Sample {
  id: string;
  filename: string;
  title: string;
  description: string;
  url: string;
  synthetic: boolean;
}

export interface RocPoint {
  fpr: number;
  tpr: number;
}

export interface SweepPoint {
  threshold: number;
  accuracy: number;
  sensitivity: number;
  specificity: number;
  f1: number;
}

export interface Metrics {
  model: string;
  split: string;
  evaluated_at: string;
  device: string;
  weights: string;
  n_images: number;
  n_positive: number;
  threshold: number;
  accuracy: number;
  auc: number;
  confusion: {
    tp: number;
    tn: number;
    fp: number;
    fn: number;
    sensitivity: number;
    specificity: number;
    precision: number;
    f1: number;
  };
  roc: { auc: number; points: RocPoint[]; best_threshold: number; best_tpr: number; best_fpr: number };
  sweep: SweepPoint[];
  localization: {
    points: { fps_per_image: number; sensitivity: number }[];
    mean_sensitivity: number;
    total_objects: number;
    n_images: number;
  };
  training_run?: Record<string, unknown>;
}
