import { useEffect, useState } from "react";
import { api, apiBaseUrl } from "./api";
import ExperimentsList from "./pages/ExperimentsList";
import ExperimentDetail from "./pages/ExperimentDetail";
import RunForm from "./pages/RunForm";
import "./index.css";

type View = { page: "list" } | { page: "detail"; id: number } | { page: "run" };

export default function App() {
  const [view, setView] = useState<View>({ page: "list" });
  // Kept in memory only: never written to storage, gone when the tab closes.
  const [token, setToken] = useState("");
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    api
      .health()
      .then((res) => setOnline(res.status === "ok"))
      .catch(() => setOnline(false));
  }, []);

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [view]);

  const statusText = online === null ? "Connecting…" : online ? "API online" : "API offline";

  return (
    <div className="app">
      <header className="topbar">
        <span className="wordmark">agent-eval</span>
        <nav aria-label="Main">
          <button
            type="button"
            className="ghost"
            aria-current={view.page !== "run" ? "page" : undefined}
            onClick={() => setView({ page: "list" })}
          >
            Experiments
          </button>
          <button
            type="button"
            className="ghost"
            aria-current={view.page === "run" ? "page" : undefined}
            onClick={() => setView({ page: "run" })}
          >
            New run
          </button>
        </nav>
        <span className="status" title={apiBaseUrl}>
          <span className={`dot ${online === null ? "" : online ? "ok" : "bad"}`} aria-hidden="true" />
          <span className="status-text">{statusText}</span>
        </span>
      </header>

      <main>
        {online === false ? (
          <p className="notice info" style={{ marginBottom: 24 }}>
            The API isn&apos;t answering. On Render&apos;s free tier it sleeps when idle and takes
            about a minute to wake up, so try again shortly.
          </p>
        ) : null}
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

      <footer>agent-eval · every number here comes from the API</footer>
    </div>
  );
}
