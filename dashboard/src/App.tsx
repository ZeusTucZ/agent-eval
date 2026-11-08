import { useEffect, useState } from "react";
import { api, apiBaseUrl } from "./api";
import ExperimentsList from "./pages/ExperimentsList";
import ExperimentDetail from "./pages/ExperimentDetail";
import RunForm from "./pages/RunForm";
import ComparePage from "./pages/ComparePage";
import LabelPage from "./pages/LabelPage";
import "./index.css";

// Routes live in the URL fragment, so every view can be linked to and the back button works,
// with no server routing (the dashboard is static files).
type View =
  | { page: "list" }
  | { page: "detail"; id: number }
  | { page: "run" }
  | { page: "compare"; a: number; b: number }
  | { page: "label"; id: number };

function parseRoute(hash: string): View {
  const parts = hash.replace(/^#\/?/, "").split("/");
  const n = (i: number) => Number(parts[i]);
  if (parts[0] === "e" && n(1)) return { page: "detail", id: n(1) };
  if (parts[0] === "compare" && n(1) && n(2)) return { page: "compare", a: n(1), b: n(2) };
  if (parts[0] === "label" && n(1)) return { page: "label", id: n(1) };
  if (parts[0] === "run") return { page: "run" };
  return { page: "list" };
}

export function go(path: string) {
  window.location.hash = `#/${path}`;
}

// `agenteval ui` hands this session's token over in the fragment (#token=...), which is
// never sent to a server. Take it, then clear it from the address bar and history. It is kept
// in sessionStorage: it survives a reload of this tab and is gone when the tab closes.
const TOKEN_KEY = "agenteval-token";

function initialToken(): string {
  const match = window.location.hash.match(/token=([^&]+)/);
  if (match) {
    window.history.replaceState(null, "", window.location.pathname + "#/");
    const token = decodeURIComponent(match[1]);
    saveToken(token);
    return token;
  }
  try {
    return window.sessionStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return ""; // storage can be blocked; the token then lasts until a reload
  }
}

function saveToken(token: string) {
  try {
    if (token) window.sessionStorage.setItem(TOKEN_KEY, token);
    else window.sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    // ignore: see initialToken
  }
}

export default function App() {
  const [token, setTokenState] = useState(initialToken);
  const setToken = (value: string) => {
    setTokenState(value);
    saveToken(value);
  };
  const [view, setView] = useState<View>(() => parseRoute(window.location.hash));
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    const onChange = () => {
      setView(parseRoute(window.location.hash));
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  useEffect(() => {
    api
      .health()
      .then((res) => setOnline(res.status === "ok"))
      .catch(() => setOnline(false));
  }, []);

  const statusText = online === null ? "Connecting…" : online ? "API online" : "API offline";
  const onRun = view.page === "run";

  return (
    <div className="app">
      <header className="topbar">
        <a className="wordmark" href="#/">agenteval</a>
        <nav aria-label="Main">
          <button type="button" className="ghost" aria-current={onRun ? undefined : "page"} onClick={() => go("")}>
            Experiments
          </button>
          <button type="button" className="ghost" aria-current={onRun ? "page" : undefined} onClick={() => go("run")}>
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
            The API isn&apos;t answering. A server on a free tier may be asleep and take about a minute to wake up;
            try again shortly.
          </p>
        ) : null}
        {view.page === "list" ? (
          <ExperimentsList />
        ) : view.page === "detail" ? (
          <ExperimentDetail id={view.id} token={token} onTokenChange={setToken} />
        ) : view.page === "compare" ? (
          <ComparePage a={view.a} b={view.b} />
        ) : view.page === "label" ? (
          <LabelPage id={view.id} token={token} onTokenChange={setToken} />
        ) : (
          <RunForm token={token} onTokenChange={setToken} />
        )}
      </main>

      <footer>agenteval · every number here comes from the API</footer>
    </div>
  );
}
