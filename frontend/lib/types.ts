// Types mirroring the FastAPI response models (src/transactionguard/api/schemas.py).

export type Role = "admin" | "analyst" | "viewer";
export type ReviewStatus = "unreviewed" | "under_review" | "resolved";
export type Assessment = "suspicious" | "not_suspicious" | "inconclusive";
export type JobStatus = "queued" | "running" | "succeeded" | "failed";
export type ScoreKind = "calibrated_probability" | "anomaly_score" | "constant_prior";
export type ContractName = "synthetic" | "creditcard";

export interface User {
  id: string;
  email: string;
  role: Role;
  workspace_id: string;
  workspace_name: string | null;
  is_active: boolean;
}

export interface Page<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}

export interface Review {
  status: ReviewStatus;
  assessment: Assessment | null;
  note: string | null;
  reviewer_id: string | null;
  updated_at: string;
}

export interface Prediction {
  id: string;
  transaction_id: string;
  contract: ContractName;
  synthetic: boolean;
  account_id: string | null;
  event_time: string | null;
  model_version: string;
  preprocessing_version: string;
  score: number;
  score_kind: ScoreKind;
  score_meaning: string;
  threshold: number;
  flagged: boolean;
  recommendation: "review" | "no_action";
  explanation_status: "not_requested" | "pending" | "ok" | "failed" | "unavailable";
  latency_ms: number;
  warnings: string[];
  batch_job_id: string | null;
  created_at: string;
  review: Review | null;
  confirmed_label: boolean | null;
  confirmed_label_source: string | null;
  idempotent_replay: boolean;
}

export interface Contribution {
  feature: string;
  value: string | number | null;
  contribution: number;
  direction: "increases_risk" | "decreases_risk" | "neutral";
  description: string;
  components: { column: string; contribution: number }[];
}

export interface Explanation {
  status: "ok" | "unavailable" | "failed";
  method: string | null;
  output_scale: string | null;
  base_value: number | null;
  output_value: number | null;
  contributions: Contribution[];
  summary: string | null;
  reason: string | null;
  additivity_error: number | null;
  disclaimer: string;
}

export interface PredictionDetail extends Prediction {
  transaction: Record<string, unknown>;
  features: Record<string, unknown>;
  explanation: Explanation | null;
  trace: Record<string, unknown>;
}

export interface FieldSpec {
  name: string;
  dtype: "float" | "int" | "category" | "datetime" | "string" | "bool";
  role: "id" | "timestamp" | "entity" | "feature" | "target";
  description: string;
  required: boolean;
  min_value: number | null;
  max_value: number | null;
  categories: string[] | null;
  unit: string | null;
}

export interface Contract {
  name: ContractName;
  provenance: "real_anonymized" | "synthetic";
  description: string;
  input_fields: FieldSpec[];
  model_features: { name: string; description: string }[];
  notes: string[];
  active_model_version: string | null;
}

export interface BatchJob {
  id: string;
  contract: ContractName;
  model_version: string;
  file_name: string;
  status: JobStatus;
  total_rows: number;
  valid_rows: number;
  invalid_rows: number;
  processed_rows: number;
  flagged_rows: number;
  attempts: number;
  error: string | null;
  validation_report: {
    issues?: { code: string; severity: string; column: string | null; count: number; message: string; example_ids: string[] }[];
    scoring?: { replayed_existing_predictions: number; n_conflicts: number };
  } | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  idempotent_replay: boolean;
}

export interface ModelSummary {
  model_version: string;
  model_name: string;
  contract: ContractName;
  score_kind: ScoreKind;
  score_meaning: string;
  threshold: number;
  is_active: boolean;
  dataset_provenance: string;
  training_session: string;
  saved_at: string;
  mlflow_run_id: string | null;
  test_metrics?: Metrics;
}

export interface Metrics {
  n: number;
  positives: number;
  prevalence: number;
  average_precision: number | null;
  average_precision_ci95?: [number, number];
  roc_auc: number | null;
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  false_positive_rate: number;
  alerts: number;
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
}

export interface EvaluationResult {
  model: string;
  model_version: string;
  score_kind: ScoreKind;
  score_meaning: string;
  params: Record<string, unknown>;
  optuna_trials: number;
  threshold: number;
  threshold_policy: Record<string, unknown>;
  validation: Metrics;
  test: Metrics;
  test_calibration: { brier: number; ece: number; bins: { bin: number; count: number; mean_predicted: number; observed_rate: number }[] } | null;
  test_pr_curve: { precision: number; recall: number }[];
  latency: { single_row_ms_p50: number; single_row_ms_p95: number; batch_rows_per_s: number; note: string } | null;
  global_importance: { feature: string; mean_abs_contribution: number }[];
  selected: boolean;
}

export interface SplitStats {
  rows: number;
  positives: number;
  prevalence: number;
}

export interface Evaluation {
  session_id: string;
  dataset: ContractName;
  synthetic: boolean;
  dataset_fingerprint: string;
  split: { config: Record<string, unknown>; train: SplitStats; valid: SplitStats; test: SplitStats };
  selection: { rule: string; selected: string; validation_ap: Record<string, number> };
  primary_metric: string;
  results: EvaluationResult[];
  test_uncertainty?: { differences: Record<string, { mean: number; ci95: [number, number]; share_positive: number }> };
  created_at: string;
}

export interface EvaluationListItem {
  id: string;
  dataset: ContractName;
  synthetic: boolean;
  session_id: string;
  selected: string;
  created_at: string;
  test_ap: Record<string, number>;
}

export interface MonitoringReport {
  id: string;
  contract: ContractName;
  model_version: string;
  status: JobStatus;
  window_start: string | null;
  window_end: string | null;
  summary: MonitoringSummary | null;
  error: string | null;
  created_at: string;
  finished_at: string | null;
}

export interface MonitoringSummary {
  status?: "drift_detected" | "no_drift_detected" | "insufficient_data";
  note?: string;
  synthetic?: boolean;
  interpretation?: string;
  reference?: { rows: number; source?: string; time_start?: number; time_end?: number; time_unit?: string };
  window?: { start: string; end: string; days: number; rows: number };
  schema_violations?: { batch_rows_total: number; batch_rows_invalid: number; issues_by_code: Record<string, number>; rejected_single_requests_422: number };
  scores?: {
    score_kind: ScoreKind;
    threshold: number;
    reference_alert_rate: number;
    current_alert_rate: number | null;
    current_alerts: number;
    reference_quantiles: Record<string, number>;
    current_quantiles: Record<string, number> | null;
  };
  alerts_by_day?: { day: string; scored: number; alerts: number }[];
  drift?: { drifted_columns?: number; drifted_share?: number; columns: Record<string, { method: string; statistic: number; statistic_kind: "p_value" | "distance"; threshold: number; drift_detected: boolean }> };
  missing_share_current?: Record<string, number>;
  missing_share_reference?: Record<string, number>;
  performance?: { available: boolean; labelled_rows: number; reason?: string; label_coverage?: number; metrics?: Metrics; caveat?: string };
}
