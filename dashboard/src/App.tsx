import { useEffect, useState } from "react";
import { api, apiBaseUrl } from "./api";
import ExperimentsList from "./pages/ExperimentsList";
import ExperimentDetail from "./pages/ExperimentDetail";
import RunForm from "./pages/RunForm";
import "./index.css";

type View = { page: "list" } | { page: "detail"; id: number } | { page: "run" };

export default function App() {
  const [view, setView] = useState<View>({ page: "list" });
  const [token, setToken] = useState("");
  const [health, setHealth] = useState<string | null>(null);

  useEffect(() => {
    api
      .health()
      .then((res) => setHealth(res.status))
      .catch(() => setHealth("unreachable"));
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand-row">
          <div className="brand-mark" aria-hidden="true">
            æ
          </div>
          <div className="brand-text">
            <p className="eyebrow">LLM agent evaluation</p>
            <h1>agent-eval</h1>
            <p className="subtitle">
              Evaluation dashboard · API: {apiBaseUrl} · status:{" "}
              <span
                className={`status-badge ${
                  health === "ok" ? "status-ok" : health ? "status-bad" : ""
                }`}
              >
                {health ?? "checking…"}
              </span>
            </p>
          </div>
        </div>
        <nav aria-label="Dashboard sections">
          <button type="button" onClick={() => setView({ page: "list" })} disabled={view.page === "list"}>
            Experiments
          </button>{" "}
          <button type="button" onClick={() => setView({ page: "run" })} disabled={view.page === "run"}>
            New run
          </button>
        </nav>
      </header>
      <main>
        {view.page === "list" ? (
          <ExperimentsList
            onOpen={(id) => setView({ page: "detail", id })}
            onNewRun={() => setView({ page: "run" })}
          />
        ) : view.page === "detail" ? (
          <ExperimentDetail
            id={view.id}
            token={token}
            onTokenChange={setToken}
            onBack={() => setView({ page: "list" })}
          />
        ) : (
          <RunForm
            token={token}
            onTokenChange={setToken}
            onExperimentReady={(id) => setView({ page: "detail", id })}
          />
        )}
      </main>
      <footer>
        <p>
          agent-eval dashboard — every number shown comes from the API; rates with no
          scores show “n/a”. The API token is kept only in this browser tab and is never
          saved in the repo.
        </p>
        <p>FastAPI + React · deployed on Render</p>
      </footer>
    </div>
  );
}
