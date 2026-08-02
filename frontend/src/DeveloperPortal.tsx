import { useCallback, useEffect, useMemo, useState } from "react";
import {
  BookOpen,
  CheckCircle2,
  CircleDot,
  Code2,
  Copy,
  ExternalLink,
  Gauge,
  Info,
  KeyRound,
  Plus,
  RefreshCw,
  Send,
  Trash2,
  WifiOff,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

type DeveloperView = "apikeys" | "docs" | "reference" | "status";
type CodeLanguage = "curl" | "javascript" | "python";

type ApiKeyRow = {
  id: string;
  name: string;
  key_prefix: string;
  created_at: string;
  expires: string | null;
  enabled: boolean;
};

type DeveloperRequestState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; statusCode: number; durationMs: number; body: Record<string, unknown> }
  | { status: "error"; message: string };

type ApiHealthDetails = {
  status: string;
  model: string;
  held_out_r2: number;
  database: string;
  cad_analysis: string;
  vision_model: string | null;
};

type DeveloperPortalProps = {
  apiUrl: string;
  sourceUrl: string;
};

const SAMPLE_REQUEST = {
  area_sqft: 1200,
  area_type: "super",
  location: "thane",
  locality: "kolshet road",
  society: "lodha amara",
  bedrooms: 2,
  bathroom: 2,
  balcony: 1,
  car_parking: 1,
  floor_num: 8,
  total_floors: 24,
  property_type: "flat",
  furnishing: "semi_furnished",
  transaction: "resale",
  ownership: "freehold",
  facing: "east",
  overlooking: "garden",
} as const;

const readable = (value: string) => value.charAt(0).toUpperCase() + value.slice(1);

function formatDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  } catch {
    return iso;
  }
}

export function DeveloperPortal({ apiUrl, sourceUrl }: DeveloperPortalProps) {
  const [view, setView] = useState<DeveloperView>("apikeys");
  const [language, setLanguage] = useState<CodeLanguage>("curl");
  const [requestBody, setRequestBody] = useState(() => JSON.stringify(SAMPLE_REQUEST, null, 2));
  const [requestState, setRequestState] = useState<DeveloperRequestState>({ status: "idle" });
  const [health, setHealth] = useState<ApiHealthDetails | null>(null);
  const [healthError, setHealthError] = useState("");
  const [copied, setCopied] = useState("");

  // API Keys state — fetched from real database
  const [apiKeys, setApiKeys] = useState<ApiKeyRow[]>([]);
  const [keysLoading, setKeysLoading] = useState(false);
  const [keysError, setKeysError] = useState("");
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [newKeyName, setNewKeyName] = useState("");
  const [createdSecret, setCreatedSecret] = useState<string | null>(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  const fetchKeys = useCallback(async () => {
    setKeysLoading(true);
    setKeysError("");
    try {
      const res = await fetch(apiUrl + "/api-keys", { cache: "no-store" });
      if (res.status === 404) {
        setKeysError("The API keys endpoint is not available yet. Redeploy the backend to enable key management.");
        return;
      }
      if (!res.ok) throw new Error("Failed to load keys (" + res.status + ")");
      const data = (await res.json()) as { keys: ApiKeyRow[] };
      setApiKeys(data.keys);
    } catch (err) {
      if (err instanceof TypeError) {
        setKeysError("Could not reach the API server.");
      } else {
        setKeysError(err instanceof Error ? err.message : "Could not load API keys");
      }
    } finally {
      setKeysLoading(false);
    }
  }, [apiUrl]);

  useEffect(() => {
    if (view === "apikeys") void fetchKeys();
  }, [fetchKeys, view]);

  const refreshHealth = async () => {
    setHealthError("");
    try {
      const response = await fetch(apiUrl + "/health", { cache: "no-store" });
      const body = (await response.json()) as ApiHealthDetails & { detail?: string };
      if (!response.ok) throw new Error(body.detail ?? "Health check failed.");
      setHealth(body);
    } catch (error) {
      setHealth(null);
      setHealthError(error instanceof Error ? error.message : "Health check failed.");
    }
  };

  useEffect(() => {
    void refreshHealth();
  }, [apiUrl]);

  const generatedCode = useMemo(() => {
    if (language === "javascript") {
      return [
        "const response = await fetch(\"" + apiUrl + "/predict\", {",
        "  method: \"POST\",",
        "  headers: { \"Content-Type\": \"application/json\" },",
        "  body: JSON.stringify(" + requestBody + "),",
        "});",
        "",
        "const prediction = await response.json();",
      ].join("\n");
    }
    if (language === "python") {
      const pythonBody = requestBody.replace(/true/g, "True").replace(/false/g, "False").replace(/null/g, "None");
      return [
        "import requests",
        "",
        "response = requests.post(",
        "    \"" + apiUrl + "/predict\",",
        "    json=" + pythonBody + ",",
        "    timeout=30,",
        ")",
        "response.raise_for_status()",
        "prediction = response.json()",
      ].join("\n");
    }
    return [
      "curl --request POST \\",
      "  --url " + apiUrl + "/predict \\",
      "  --header 'Content-Type: application/json' \\",
      "  --data '" + requestBody + "'",
    ].join("\n");
  }, [apiUrl, language, requestBody]);

  const copyText = async (label: string, value: string) => {
    await navigator.clipboard.writeText(value);
    setCopied(label);
    window.setTimeout(() => setCopied(""), 1600);
  };

  const sendDeveloperRequest = async () => {
    setRequestState({ status: "loading" });
    try {
      const parsed = JSON.parse(requestBody) as Record<string, unknown>;
      const startedAt = performance.now();
      const response = await fetch(apiUrl + "/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(parsed),
      });
      const body = (await response.json()) as Record<string, unknown>;
      const durationMs = Math.round(performance.now() - startedAt);
      if (!response.ok) {
        throw new Error(typeof body.detail === "string" ? body.detail : "Request failed with " + response.status + ".");
      }
      setRequestState({ status: "success", statusCode: response.status, durationMs, body });
    } catch (error) {
      setRequestState({ status: "error", message: error instanceof Error ? error.message : "The request failed." });
    }
  };

  // Real API key CRUD
  const handleCreateKey = async (e: React.FormEvent) => {
    e.preventDefault();
    const name = newKeyName.trim() || "API Key";
    try {
      const res = await fetch(apiUrl + "/api-keys", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!res.ok) throw new Error("Failed to create key");
      const data = (await res.json()) as { key: string };
      setCreatedSecret(data.key);
      setIsCreateModalOpen(false);
      setNewKeyName("");
      void fetchKeys();
    } catch {
      // silent — modal stays open
    }
  };

  const handleToggleKey = async (id: string, currentlyEnabled: boolean) => {
    // Optimistic update
    setApiKeys((prev) => prev.map((k) => (k.id === id ? { ...k, enabled: !currentlyEnabled } : k)));
    try {
      await fetch(apiUrl + "/api-keys/" + id + "?enabled=" + (!currentlyEnabled), { method: "PATCH" });
    } catch {
      setApiKeys((prev) => prev.map((k) => (k.id === id ? { ...k, enabled: currentlyEnabled } : k)));
    }
  };

  const handleDeleteKey = async (id: string) => {
    setDeleteConfirmId(null);
    setApiKeys((prev) => prev.filter((k) => k.id !== id));
    try {
      await fetch(apiUrl + "/api-keys/" + id, { method: "DELETE" });
    } catch {
      void fetchKeys();
    }
  };

  const endpointRows = [
    { method: "GET", path: "/health", summary: "Service, model, database, and CAD readiness" },
    { method: "POST", path: "/predict", summary: "Run the 90.64% held-out R² price ensemble" },
    { method: "POST", path: "/analyze", summary: "Analyze a floor plan and predict its property value" },
    { method: "GET", path: "/project/{id}", summary: "Load a synchronized property project" },
    { method: "PUT", path: "/project/{id}", summary: "Create or update a synchronized project" },
    { method: "GET", path: "/project/{id}/predictions", summary: "List stored prediction traces" },
    { method: "POST", path: "/api-keys", summary: "Create a new API key" },
    { method: "GET", path: "/api-keys", summary: "List all API keys" },
    { method: "PATCH", path: "/api-keys/{id}", summary: "Enable or disable an API key" },
    { method: "DELETE", path: "/api-keys/{id}", summary: "Permanently delete an API key" },
  ] as const;

  return (
    <section className="developer-page developer-page-compact">
      <aside className="developer-sidebar card">
        <header>
          <span className="eyebrow">Build with SpaceWorth</span>
          <h1>API</h1>
        </header>
        <nav aria-label="Developer sections">
          <button className={view === "apikeys" ? "active" : ""} onClick={() => setView("apikeys")}>
            <KeyRound /><span><strong>API Keys</strong><small>Manage access</small></span>
          </button>
          <button className={view === "docs" ? "active" : ""} onClick={() => setView("docs")}>
            <BookOpen /><span><strong>Docs</strong><small>Try requests</small></span>
          </button>
          <button className={view === "reference" ? "active" : ""} onClick={() => setView("reference")}>
            <Code2 /><span><strong>Reference</strong><small>All endpoints</small></span>
          </button>
          <button className={view === "status" ? "active" : ""} onClick={() => setView("status")}>
            <Gauge /><span><strong>Status</strong><small>Live health</small></span>
          </button>
        </nav>
      </aside>

      <div className="developer-content card">
        {/* API KEYS — real database */}
        {view === "apikeys" && (
          <div className="developer-keys-section">
            <header className="developer-content-header">
              <div>
                <span className="eyebrow">Access management</span>
                <h2>API Keys</h2>
                <p>Create and manage keys for the SpaceWorth API. Keys are stored in the database.</p>
              </div>
              <Button className="dark-button create-key-btn" onClick={() => setIsCreateModalOpen(true)}>
                <Plus /> Create Key
              </Button>
            </header>

            {keysError && <p className="developer-request-error">{keysError}</p>}

            <div className="developer-keys-table-wrap">
              <table className="developer-keys-table">
                <thead>
                  <tr>
                    <th>NAME</th>
                    <th>KEY</th>
                    <th>CREATED</th>
                    <th>EXPIRES</th>
                    <th>ENABLED</th>
                    <th className="align-right"></th>
                  </tr>
                </thead>
                <tbody>
                  {keysLoading && apiKeys.length === 0 && (
                    <tr><td colSpan={6} className="empty-table-cell">Loading keys…</td></tr>
                  )}
                  {!keysLoading && apiKeys.length === 0 && !keysError && (
                    <tr><td colSpan={6} className="empty-table-cell">No API keys yet. Click <strong>+ Create Key</strong> to get started.</td></tr>
                  )}
                  {apiKeys.map((item) => (
                    <tr key={item.id} className={!item.enabled ? "disabled-row" : ""}>
                      <td className="key-name-cell">
                        <span>{item.name}</span>
                        <span className="info-icon-wrapper" title={`ID: ${item.id}`}><Info /></span>
                      </td>
                      <td className="key-code-cell">
                        <code>••••••••••••••••{item.key_prefix}</code>
                      </td>
                      <td className="key-date-cell">{formatDate(item.created_at)}</td>
                      <td className="key-expires-cell">{item.expires ? formatDate(item.expires) : "Never"}</td>
                      <td className="key-toggle-cell">
                        <button
                          type="button"
                          className={`key-switch ${item.enabled ? "on" : "off"}`}
                          onClick={() => void handleToggleKey(item.id, item.enabled)}
                          aria-label={`Toggle ${item.name}`}
                        >
                          <span className="switch-thumb" />
                        </button>
                      </td>
                      <td className="key-actions-cell align-right">
                        <Button variant="ghost" size="icon" className="delete-key-btn" onClick={() => setDeleteConfirmId(item.id)} title="Delete">
                          <Trash2 />
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* DOCS */}
        {view === "docs" && (
          <>
            <header className="developer-content-header">
              <div>
                <span className="eyebrow">Request builder</span>
                <h2>Predict a price</h2>
                <p>Edit the JSON payload, send it to the live API, and inspect the response.</p>
              </div>
              <span className={"developer-live-badge " + (health ? "online" : "")}><i />{health ? "Online" : "Checking"}</span>
            </header>
            <section className="developer-endpoint">
              <span className="method post">POST</span>
              <code>{apiUrl}/predict</code>
              <button onClick={() => void copyText("url", apiUrl + "/predict")} aria-label="Copy endpoint URL">
                {copied === "url" ? <CheckCircle2 /> : <Copy />}
              </button>
            </section>
            <div className="developer-builder">
              <section>
                <div className="developer-section-title">
                  <div><span>Request body</span><small>application/json</small></div>
                  <button onClick={() => setRequestBody(JSON.stringify(SAMPLE_REQUEST, null, 2))}><RefreshCw /> Reset</button>
                </div>
                <textarea aria-label="Prediction request JSON" spellCheck={false} value={requestBody} onChange={(event) => setRequestBody(event.target.value)} />
              </section>
              <section>
                <div className="developer-section-title">
                  <div><span>Code example</span><small>Ready to paste</small></div>
                  <button onClick={() => void copyText("code", generatedCode)}>
                    {copied === "code" ? <CheckCircle2 /> : <Copy />} {copied === "code" ? "Copied" : "Copy"}
                  </button>
                </div>
                <div className="developer-language-tabs">
                  {(["curl", "javascript", "python"] as CodeLanguage[]).map((item) => (
                    <button key={item} className={language === item ? "active" : ""} onClick={() => setLanguage(item)}>
                      {item === "javascript" ? "JavaScript" : readable(item)}
                    </button>
                  ))}
                </div>
                <pre><code>{generatedCode}</code></pre>
              </section>
            </div>
            <section className="developer-response">
              <div className="developer-section-title">
                <div>
                  <span>Response</span>
                  <small>{requestState.status === "success" ? requestState.statusCode + " OK · " + requestState.durationMs + " ms" : "Railway API"}</small>
                </div>
                <Button className="dark-button" onClick={() => void sendDeveloperRequest()} disabled={requestState.status === "loading"}>
                  <Send />{requestState.status === "loading" ? "Sending…" : "Send"}
                </Button>
              </div>
              {requestState.status === "success" ? (
                <pre><code>{JSON.stringify(requestState.body, null, 2)}</code></pre>
              ) : requestState.status === "error" ? (
                <p className="developer-request-error">{requestState.message}</p>
              ) : (
                <div className="developer-response-empty"><CircleDot /><span>Response will appear here.</span></div>
              )}
            </section>
          </>
        )}

        {/* REFERENCE */}
        {view === "reference" && (
          <>
            <header className="developer-content-header">
              <div>
                <span className="eyebrow">v2.0.0</span>
                <h2>API reference</h2>
                <p>All routes from the live OpenAPI document.</p>
              </div>
              <a className="developer-header-link" href={apiUrl + "/redoc"} target="_blank" rel="noreferrer">ReDoc <ExternalLink /></a>
            </header>
            <div className="developer-endpoint-list">
              {endpointRows.map((ep) => (
                <article key={ep.method + ep.path}>
                  <span className={"method " + ep.method.toLowerCase()}>{ep.method}</span>
                  <div><code>{ep.path}</code><p>{ep.summary}</p></div>
                  <a href={apiUrl + "/docs"} target="_blank" rel="noreferrer" aria-label={"Open " + ep.path}><ExternalLink /></a>
                </article>
              ))}
            </div>
          </>
        )}

        {/* STATUS */}
        {view === "status" && (
          <>
            <header className="developer-content-header">
              <div>
                <span className="eyebrow">Production</span>
                <h2>Live status</h2>
                <p>Fetched from the running Railway service.</p>
              </div>
              <Button variant="outline" onClick={() => void refreshHealth()}><RefreshCw /> Refresh</Button>
            </header>
            {health ? (
              <div className="developer-status-grid">
                <article><span>Service</span><strong>{health.status}</strong><small>Health check passed</small></article>
                <article><span>Model</span><strong>{(health.held_out_r2 * 100).toFixed(2)}% R²</strong><small>{health.model}</small></article>
                <article><span>Database</span><strong>{health.database}</strong><small>PostgreSQL</small></article>
                <article><span>CAD</span><strong>{health.cad_analysis}</strong><small>{health.vision_model ?? "—"}</small></article>
              </div>
            ) : (
              <div className="developer-status-error"><WifiOff /><strong>Offline</strong><p>{healthError || "Checking…"}</p></div>
            )}
          </>
        )}
      </div>

      {/* CREATE KEY MODAL */}
      {isCreateModalOpen && (
        <div className="developer-modal-backdrop" onClick={() => setIsCreateModalOpen(false)}>
          <div className="developer-modal card" onClick={(e) => e.stopPropagation()}>
            <header className="developer-modal-header">
              <h3>Create API key</h3>
              <button type="button" onClick={() => setIsCreateModalOpen(false)} aria-label="Close"><X /></button>
            </header>
            <form onSubmit={(e) => void handleCreateKey(e)} className="developer-modal-form">
              <div className="form-group">
                <label htmlFor="key-name-input">Name</label>
                <Input id="key-name-input" placeholder="e.g. Production, Staging" value={newKeyName} onChange={(e) => setNewKeyName(e.target.value)} autoFocus />
              </div>
              <div className="developer-modal-actions">
                <Button type="button" variant="outline" onClick={() => setIsCreateModalOpen(false)}>Cancel</Button>
                <Button type="submit" className="dark-button">Create</Button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* SECRET DISPLAY MODAL */}
      {createdSecret && (
        <div className="developer-modal-backdrop" onClick={() => setCreatedSecret(null)}>
          <div className="developer-modal card secret-modal" onClick={(e) => e.stopPropagation()}>
            <header className="developer-modal-header">
              <h3>Save your key</h3>
              <button type="button" onClick={() => setCreatedSecret(null)} aria-label="Close"><X /></button>
            </header>
            <div className="developer-modal-body">
              <div className="secret-warning-box">
                <KeyRound />
                <p><strong>Copy this key now.</strong> You will not be able to see the full secret again.</p>
              </div>
              <div className="secret-key-display">
                <code>{createdSecret}</code>
                <Button variant="outline" onClick={() => void copyText("secret-modal", createdSecret)}>
                  {copied === "secret-modal" ? <CheckCircle2 /> : <Copy />} {copied === "secret-modal" ? "Copied" : "Copy"}
                </Button>
              </div>
              <div className="developer-modal-actions">
                <Button className="dark-button" onClick={() => setCreatedSecret(null)}>Done</Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* DELETE CONFIRM */}
      {deleteConfirmId && (
        <div className="developer-modal-backdrop" onClick={() => setDeleteConfirmId(null)}>
          <div className="developer-modal card" onClick={(e) => e.stopPropagation()}>
            <header className="developer-modal-header">
              <h3>Delete key</h3>
              <button type="button" onClick={() => setDeleteConfirmId(null)} aria-label="Close"><X /></button>
            </header>
            <div className="developer-modal-body">
              <p>This key will be permanently deleted. Applications using it will immediately lose access.</p>
              <div className="developer-modal-actions">
                <Button variant="outline" onClick={() => setDeleteConfirmId(null)}>Cancel</Button>
                <Button className="destructive-button" onClick={() => void handleDeleteKey(deleteConfirmId)}>Delete</Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
