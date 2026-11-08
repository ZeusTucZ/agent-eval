// The ONLY place that knows the API base URL. It comes from VITE_API_URL,
// baked in at build time (Render Static Site env var); locally it defaults
// to the FastAPI dev server. No other file reads import.meta.env.
//
// Endpoints exposed here are exactly the 7 the API defines (agenteval/api.py):
//   GET  /health
//   GET  /api/experiments
//   GET  /api/experiments/{id}
//   GET  /api/experiments/{id}/trials
//   POST /api/runs                 (X-API-Token required)
//   GET  /api/runs/{id}
//   POST /api/experiments/{id}/rescore  (X-API-Token required)
import type { ExperimentSummary, Job, RunRequest, Trial } from "./types";

const API_URL = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "")
  || "http://localhost:8000";

export const apiBaseUrl = API_URL;

interface ApiError {
  detail?: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, init);
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const body = (await res.json()) as ApiError;
      if (body.detail) message = body.detail;
    } catch {
      // Keep the HTTP status text when the error body is not JSON.
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health(): Promise<{ status: string }> {
    return request("/health");
  },

  listExperiments(): Promise<ExperimentSummary[]> {
    return request("/api/experiments");
  },

  getExperiment(id: number): Promise<ExperimentSummary> {
    return request(`/api/experiments/${id}`);
  },

  listTrials(id: number): Promise<{ experiment_id: number; trials: Trial[] }> {
    return request(`/api/experiments/${id}/trials`);
  },

  createRun(body: RunRequest, token: string): Promise<Job> {
    return request("/api/runs", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Token": token,
      },
      body: JSON.stringify(body),
    });
  },

  getRun(jobId: string): Promise<Job> {
    return request(`/api/runs/${jobId}`);
  },

  rescoreExperiment(id: number, token: string): Promise<ExperimentSummary> {
    return request(`/api/experiments/${id}/rescore`, {
      method: "POST",
      headers: { "X-API-Token": token },
    });
  },
};
