// Types mirror exactly what agenteval/api.py returns. No field here is
// invented: if the API does not return it, it is not rendered as if it did.
// See experiment_summary() in agenteval/api.py and Trial/Span/Score in
// agenteval/models.py. Note Trial.model_dump() does not include the
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
  pass_rate: number | null; // null when no case has an expected answer
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
    // Only from API versions with retrieval scoring; absent on older deployments.
    relevant_found?: number | null;
  };
  // Retrieval runs only: correct / answer_failure / retrieval_failure / unsupported_correct.
  retrieval_diagnosis?: Record<string, number>;
  failures: Record<string, number>;
  // From agenteval 1.0 on; optional so the dashboard still reads older servers.
  status?: "running" | "completed" | "interrupted";
  planned_trials?: number | null;
  environment?: string;
  by_category?: Record<string, { pass_rate: number; trials: number }>;
  other_scores?: Record<string, { mean: number; trials: number }>;
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
  // The question it answered; null when the server can't find the dataset.
  case?: { input: string; expected_output: string; category: string } | null;
}

export interface CaseChange {
  case_id: number;
  input: string;
  category: string;
  expected: string;
  before: string | null;
  after: string | null;
  before_rate: number;
  after_rate: number;
}

export interface Interval {
  before: number;
  after: number;
  change: number;
  low: number;
  high: number;
}

export interface Comparison {
  a: number;
  b: number;
  errors: string[];
  warnings: string[];
  changes: string[];
  verdict: "Better" | "Worse" | "Trade-off" | "Inconclusive" | null;
  reason?: string;
  cases_compared?: number;
  p_value?: number;
  pass_rate?: Interval;
  latency_ms?: Interval;
  cost?: Interval;
  categories?: Record<string, { before: number; after: number; cases: number }>;
  improved?: CaseChange[];
  regressed?: CaseChange[];
  safety_regressions?: CaseChange[];
  partly_changed?: number;
}

export interface Rubric {
  name: string;
  question: string;
  uses_expected: boolean;
  uses_sources: boolean;
  uses_answer: boolean;
}

export interface LabelQueue {
  rubric: string;
  labelled: number;
  judged: number;
  agreement: {
    compared: number;
    agreement: number;
    kappa: number;
    reading: string;
    human_pass_rate: number;
    judge_pass_rate: number;
  } | null;
  next: {
    trial_id: number;
    question: string | null;
    expected: string | null;
    sources: string;
    answer: string | null;
  }[];
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
  dataset: "arithmetic" | "mixed" | "hard" | "rag";
  attempts: number;
  model?: string;
  name: string;
  cases?: number;
  seed?: number;
  // rag dataset only
  top_k?: number;
  embedding?: "minilm" | "hashing";
}
