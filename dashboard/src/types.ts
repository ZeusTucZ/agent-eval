// Types mirror exactly what evalkit/api.py returns. No field here is
// invented: if the API does not return it, it is not rendered as if it did.
// See experiment_summary() in evalkit/api.py and Trial/Span/Score in
// evalkit/models.py. Note Trial.model_dump() does not include the
// duration_ms / working_ms properties, so durations are derived from the
// returned spans (max end - min start) where shown.

export interface ExperimentSummary {
  id: number;
  name: string;
  dataset_name: string;
  dataset_version: string | null;
  model: string;
  config_version: string;
  attempts: number;
  created_at: string;
  trial_count: number;
  pass_rate: number;
  latency_ms: { p50: number; p95: number };
  latency_working_ms: { p50: number; p95: number };
  cost: number;
  cost_per_success: number | null;
  tool_rates: {
    right_tool: number | null;
    valid_arguments: number | null;
    ran_without_error: number | null;
    backed_by_tools: number | null;
    no_injected_text: number | null;
  };
  failures: Record<string, number>;
}

export interface Span {
  trial_id: number;
  kind: "llm" | "tool";
  name: string;
  start: number;
  end: number;
  status: "running" | "success" | "error";
  error_type: string | null;
  error_message: string | null;
  input: string | null;
  output: string | null;
  input_tokens: number;
  output_tokens: number;
  retry: number;
}

export interface Score {
  scorer: string;
  value: number;
  reasoning: string | null;
}

export interface Trial {
  id: number;
  case_id: number;
  attempt: number;
  output: string | null;
  failure_type: string | null;
  spans: Span[];
  scores: Score[];
}

export type JobStatus = "queued" | "running" | "completed" | "failed";

export interface Job {
  id: string;
  status: JobStatus;
  agent: string;
  dataset: string;
  attempts: number;
  model: string | null;
  name: string;
  experiment_id: number | null;
  error: string | null;
  created_at: string;
}

export interface RunRequest {
  agent: "mock" | "groq" | "gemini";
  dataset: "arithmetic" | "mixed" | "hard";
  attempts: number;
  model?: string;
  name: string;
  cases?: number;
  seed?: number;
}
