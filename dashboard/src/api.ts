// The only place that knows where the API is. VITE_API_URL is baked in at build time for a
// separate deployment (a Render Static Site); without it the dashboard talks to the server
// that served it, which is how `agenteval ui` runs. In development Vite proxies /api to
// the FastAPI server on :8000 (see vite.config.ts).
import type {
  Comparison,
  ExperimentSummary,
  Job,
  LabelQueue,
  Rubric,
  RunOptions,
  RunRequest,
  Trial,
  TrendPoint,
} from "./types";

const API_URL = ((import.meta.env.VITE_API_URL as string | undefined) ?? "").replace(/\/$/, "");

export const apiBaseUrl = API_URL || window.location.origin;

interface ApiError {
  detail?: string | { msg?: string }[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as ApiError;
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail) && body.detail[0]?.msg) message = body.detail[0].msg;
    } catch {
      // Keep the HTTP status text when the error body is not JSON.
    }
    throw new Error(message);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

function withToken(token: string, init: RequestInit = {}): RequestInit {
  return { ...init, headers: { ...(init.headers ?? {}), "X-API-Token": token, "Content-Type": "application/json" } };
}

export const PAGE_SIZE = 50;

export const api = {
  health(): Promise<{ status: string }> {
    return request("/health");
  },

  listExperiments(offset = 0): Promise<ExperimentSummary[]> {
    return request(`/api/experiments?limit=${PAGE_SIZE}&offset=${offset}`);
  },

  getExperiment(id: number): Promise<ExperimentSummary> {
    return request(`/api/experiments/${id}`);
  },

  listTrials(id: number): Promise<{ experiment_id: number; trials: Trial[] }> {
    return request(`/api/experiments/${id}/trials`);
  },

  createRun(body: RunRequest, token: string): Promise<Job> {
    return request("/api/runs", withToken(token, { method: "POST", body: JSON.stringify(body) }));
  },

  runOptions(): Promise<RunOptions> {
    return request("/api/run-options");
  },

  getRun(jobId: string): Promise<Job> {
    return request(`/api/runs/${jobId}`);
  },

  rescoreExperiment(id: number, token: string): Promise<ExperimentSummary> {
    return request(`/api/experiments/${id}/rescore`, withToken(token, { method: "POST" }));
  },

  renameExperiment(id: number, name: string, token: string): Promise<ExperimentSummary> {
    return request(`/api/experiments/${id}`, withToken(token, { method: "PATCH", body: JSON.stringify({ name }) }));
  },

  deleteExperiment(id: number, token: string): Promise<void> {
    return request(`/api/experiments/${id}`, withToken(token, { method: "DELETE" }));
  },

  compare(a: number, b: number): Promise<Comparison> {
    return request(`/api/compare?a=${a}&b=${b}`);
  },

  trends(dataset: string, model: string): Promise<TrendPoint[]> {
    return request(`/api/trends?dataset=${encodeURIComponent(dataset)}&model=${encodeURIComponent(model)}`);
  },

  rubrics(): Promise<Rubric[]> {
    return request("/api/rubrics");
  },

  labels(id: number, rubric: string): Promise<LabelQueue> {
    return request(`/api/experiments/${id}/labels?rubric=${encodeURIComponent(rubric)}`);
  },

  addLabel(id: number, trialId: number, rubric: string, passed: boolean, token: string): Promise<void> {
    return request(`/api/experiments/${id}/labels`, withToken(token, {
      method: "POST",
      body: JSON.stringify({ trial_id: trialId, rubric, passed }),
    }));
  },
};
