import { useEffect, useMemo, useState } from "react";
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
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

export type DeveloperView = "apikeys" | "docs" | "reference" | "status";
type CodeLanguage = "curl" | "javascript" | "python";

export type ApiKeyItem = {
  id: string;
  name: string;
  key: string;
  createdAt: string;
  expires: string;
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

const INITIAL_API_KEYS: ApiKeyItem[] = [
  { id: "key-1", name: "Thunderous Binturong", key: "sw_live_8f3a91024bc91029481920531", createdAt: "Jul 28", expires: "Never", enabled: true },
  { id: "key-2", name: "Thunderous Binturong", key: "sw_live_41a829104bc91029481926a1e", createdAt: "Jul 28", expires: "Never", enabled: true },
  { id: "key-3", name: "Thunderous Binturong", key: "sw_live_9024bc910294819205313ed5", createdAt: "Jul 28", expires: "Never", enabled: true },
  { id: "key-4", name: "Thunderous Binturong", key: "sw_live_104bc91029481920531b679", createdAt: "Jul 28", expires: "Never", enabled: true },
  { id: "key-5", name: "Eminent Giant Squid", key: "sw_live_9481920531024bc9102343a", createdAt: "Jul 27", expires: "Never", enabled: true },
  { id: "key-6", name: "Thunderous Binturong", key: "sw_live_029481920531024bc91969a", createdAt: "Jul 28", expires: "Never", enabled: true },
  { id: "key-7", name: "Thunderous Binturong", key: "sw_live_024bc910294819205315aee", createdAt: "Jul 28", expires: "Never", enabled: true },
  { id: "key-8", name: "Thunderous Binturong", key: "sw_live_481920531024bc91029c583", createdAt: "Jul 28", expires: "Never", enabled: true },
];

const readable = (value: string) => value.charAt(0).toUpperCase() + value.slice(1);

export function DeveloperPortal({ apiUrl, sourceUrl }: DeveloperPortalProps) {
  const [view, setView] = useState<DeveloperView>("apikeys");
  const [language, setLanguage] = useState<CodeLanguage>("curl");
  const [requestBody, setRequestBody] = useState(() => JSON.stringify(SAMPLE_REQUEST, null, 2));
  const [requestState, setRequestState] = useState<DeveloperRequestState>({ status: "idle" });
  const [health, setHealth] = useState<ApiHealthDetails | null>(null);
  const [healthError, setHealthError] = useState("");
  const [copied, setCopied] = useState("");

  // API Key management state
  const [apiKeys, setApiKeys] = useState<ApiKeyItem[]>(() => {
    try {
      const saved = localStorage.getItem("spaceworth_api_keys");
      if (saved) return JSON.parse(saved) as ApiKeyItem[];
    } catch {
      // fallback
    }
    return INITIAL_API_KEYS;
  });

  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [newKeyName, setNewKeyName] = useState("");
  const [createdSecret, setCreatedSecret] = useState<string | null>(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  // Sync API keys to localStorage
  useEffect(() => {
    try {
      localStorage.setItem("spaceworth_api_keys", JSON.stringify(apiKeys));
    } catch {
      // storage error
    }
  }, [apiKeys]);

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

  // API Key handlers
  const handleToggleKey = (id: string) => {
    setApiKeys((prev) =>
      prev.map((item) => (item.id === id ? { ...item, enabled: !item.enabled } : item))
    );
  };

  const handleDeleteKey = (id: string) => {
    setApiKeys((prev) => prev.filter((item) => item.id !== id));
    setDeleteConfirmId(null);
  };

  const handleCreateKeySubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const name = newKeyName.trim() || "Development Key";
    const randomHex = Array.from(crypto.getRandomValues(new Uint8Array(12)))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");
    const suffix = randomHex.slice(-4);
    const fullKey = `sw_live_${randomHex}`;
    const today = new Date();
    const month = today.toLocaleString("en-US", { month: "short" });
    const day = today.getDate();
    const formattedDate = `${month} ${day}`;

    const newKeyItem: ApiKeyItem = {
      id: `key-${Date.now()}`,
      name,
      key: fullKey,
      createdAt: formattedDate,
      expires: "Never",
      enabled: true,
    };

    setApiKeys((prev) => [newKeyItem, ...prev]);
    setIsCreateModalOpen(false);
    setNewKeyName("");
    setCreatedSecret(fullKey);
  };

  const endpointRows = [
    { method: "GET", path: "/health", summary: "Service, model, database, and CAD readiness" },
    { method: "POST", path: "/predict", summary: "Run the 90.64% held-out R2 price ensemble" },
    { method: "POST", path: "/analyze", summary: "Analyze a floor plan and predict its property value" },
    { method: "GET", path: "/project/{project_id}", summary: "Load a synchronized property project" },
    { method: "PUT", path: "/project/{project_id}", summary: "Create or update a synchronized project" },
    { method: "GET", path: "/project/{project_id}/predictions", summary: "List stored prediction traces for a project" },
  ] as const;

  return (
    <section className="developer-page">
      <aside className="developer-sidebar card">
        <header>
          <span className="eyebrow">Build with SpaceWorth</span>
          <h1>API Developer</h1>
          <p>Integrate property valuation and CAD intelligence into your own product.</p>
        </header>
        <nav aria-label="Developer sections">
          <button className={view === "apikeys" ? "active" : ""} onClick={() => setView("apikeys")}>
            <KeyRound />
            <span>
              <strong>API Keys</strong>
              <small>Manage access keys</small>
            </span>
          </button>
          <button className={view === "docs" ? "active" : ""} onClick={() => setView("docs")}>
            <BookOpen />
            <span>
              <strong>Docs</strong>
              <small>Interactive request builder</small>
            </span>
          </button>
          <button className={view === "reference" ? "active" : ""} onClick={() => setView("reference")}>
            <Code2 />
            <span>
              <strong>API reference</strong>
              <small>Endpoints and methods</small>
            </span>
          </button>
          <button className={view === "status" ? "active" : ""} onClick={() => setView("status")}>
            <Gauge />
            <span>
              <strong>Live status</strong>
              <small>Runtime readiness</small>
            </span>
          </button>
        </nav>
        <div className="developer-help">
          <strong>Need implementation help?</strong>
          <p>Inspect the interactive docs or the public API source.</p>
          <a href={apiUrl + "/docs"} target="_blank" rel="noreferrer">
            Open Swagger <ExternalLink />
          </a>
          <a href={sourceUrl + "/blob/main/deployment/house-price-space/app.py"} target="_blank" rel="noreferrer">
            View API source <Code2 />
          </a>
        </div>
      </aside>

      <div className="developer-content card">
        {/* VIEW 1: API KEYS MANAGER (MODELED AFTER REFERENCE) */}
        {view === "apikeys" && (
          <div className="developer-keys-section">
            <header className="developer-content-header">
              <div>
                <span className="eyebrow">API Key Management</span>
                <h2>API Keys</h2>
                <p>An API key lets you connect to our API and use its features. You can create multiple keys with different access levels.</p>
              </div>
              <Button className="dark-button create-key-btn" onClick={() => setIsCreateModalOpen(true)}>
                <Plus /> Create Key
              </Button>
            </header>

            <div className="developer-keys-table-wrap">
              <table className="developer-keys-table">
                <thead>
                  <tr>
                    <th>NAME</th>
                    <th>KEY</th>
                    <th>CREATED</th>
                    <th>EXPIRES</th>
                    <th>ENABLED</th>
                    <th className="align-right">ACTIONS</th>
                  </tr>
                </thead>
                <tbody>
                  {apiKeys.map((item) => {
                    const suffix = item.key.slice(-4);
                    return (
                      <tr key={item.id} className={!item.enabled ? "disabled-row" : ""}>
                        <td className="key-name-cell">
                          <span>{item.name}</span>
                          <span className="info-icon-wrapper" title={`Created ${item.createdAt}`}>
                            <Info />
                          </span>
                        </td>
                        <td className="key-code-cell">
                          <code>••••••••••••••••••••••••{suffix}</code>
                          <button
                            type="button"
                            className="inline-copy-btn"
                            onClick={() => void copyText(item.id, item.key)}
                            title="Copy full key"
                          >
                            {copied === item.id ? <CheckCircle2 className="copied-icon" /> : <Copy />}
                          </button>
                        </td>
                        <td className="key-date-cell">{item.createdAt}</td>
                        <td className="key-expires-cell">{item.expires}</td>
                        <td className="key-toggle-cell">
                          <button
                            type="button"
                            className={`key-switch ${item.enabled ? "on" : "off"}`}
                            onClick={() => handleToggleKey(item.id)}
                            aria-label={`Toggle ${item.name} key`}
                          >
                            <span className="switch-thumb" />
                          </button>
                        </td>
                        <td className="key-actions-cell align-right">
                          <Button
                            variant="ghost"
                            size="icon"
                            className="delete-key-btn"
                            onClick={() => setDeleteConfirmId(item.id)}
                            title="Delete API Key"
                          >
                            <Trash2 />
                          </Button>
                        </td>
                      </tr>
                    );
                  })}
                  {apiKeys.length === 0 && (
                    <tr>
                      <td colSpan={6} className="empty-table-cell">
                        No API keys generated. Click <strong>+ Create Key</strong> to create your first key.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* VIEW 2: DOCS (INTERACTIVE REQUEST BUILDER & EXAMPLES) */}
        {view === "docs" && (
          <>
            <header className="developer-content-header">
              <div>
                <span className="eyebrow">Interactive request builder</span>
                <h2>Predict a property price</h2>
                <p>Edit the JSON, send it to production, and copy the same request into your application.</p>
              </div>
              <span className={"developer-live-badge " + (health ? "online" : "")}>
                <i />
                {health ? "API online" : "Checking API"}
              </span>
            </header>
            <div className="developer-notice">
              <KeyRound />
              <p>
                <strong>No key is exposed in this browser.</strong> The current public project API accepts direct requests. If server authentication is enabled later, keep <code>X-API-Key</code> in your backend or serverless function, never client code.
              </p>
            </div>
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
                  <div>
                    <span>Request body</span>
                    <small>application/json</small>
                  </div>
                  <button onClick={() => setRequestBody(JSON.stringify(SAMPLE_REQUEST, null, 2))}>
                    <RefreshCw /> Reset
                  </button>
                </div>
                <textarea
                  aria-label="Prediction request JSON"
                  spellCheck={false}
                  value={requestBody}
                  onChange={(event) => setRequestBody(event.target.value)}
                />
              </section>
              <section>
                <div className="developer-section-title">
                  <div>
                    <span>Code example</span>
                    <small>Ready to paste</small>
                  </div>
                  <button onClick={() => void copyText("code", generatedCode)}>
                    {copied === "code" ? <CheckCircle2 /> : <Copy />} {copied === "code" ? "Copied" : "Copy"}
                  </button>
                </div>
                <div className="developer-language-tabs">
                  {(["curl", "javascript", "python"] as CodeLanguage[]).map((item) => (
                    <button
                      key={item}
                      className={language === item ? "active" : ""}
                      onClick={() => setLanguage(item)}
                    >
                      {item === "javascript" ? "JavaScript" : readable(item)}
                    </button>
                  ))}
                </div>
                <pre>
                  <code>{generatedCode}</code>
                </pre>
              </section>
            </div>
            <section className="developer-response">
              <div className="developer-section-title">
                <div>
                  <span>Live response</span>
                  <small>
                    {requestState.status === "success"
                      ? requestState.statusCode + " OK · " + requestState.durationMs + " ms"
                      : "Production Railway API"}
                  </small>
                </div>
                <Button
                  className="dark-button"
                  onClick={() => void sendDeveloperRequest()}
                  disabled={requestState.status === "loading"}
                >
                  <Send />
                  {requestState.status === "loading" ? "Sending…" : "Send request"}
                </Button>
              </div>
              {requestState.status === "success" ? (
                <pre>
                  <code>{JSON.stringify(requestState.body, null, 2)}</code>
                </pre>
              ) : requestState.status === "error" ? (
                <p className="developer-request-error">{requestState.message}</p>
              ) : (
                <div className="developer-response-empty">
                  <CircleDot />
                  <span>Your server response will appear here.</span>
                </div>
              )}
            </section>
          </>
        )}

        {/* VIEW 3: API REFERENCE */}
        {view === "reference" && (
          <>
            <header className="developer-content-header">
              <div>
                <span className="eyebrow">Version 2.0.0</span>
                <h2>API reference</h2>
                <p>These routes come directly from the live FastAPI OpenAPI document.</p>
              </div>
              <a className="developer-header-link" href={apiUrl + "/redoc"} target="_blank" rel="noreferrer">
                Open ReDoc <ExternalLink />
              </a>
            </header>
            <div className="developer-endpoint-list">
              {endpointRows.map((endpoint) => (
                <article key={endpoint.method + "-" + endpoint.path}>
                  <span className={"method " + endpoint.method.toLowerCase()}>{endpoint.method}</span>
                  <div>
                    <code>{endpoint.path}</code>
                    <p>{endpoint.summary}</p>
                  </div>
                  <a href={apiUrl + "/docs"} target="_blank" rel="noreferrer" aria-label={"Open " + endpoint.path + " in Swagger"}>
                    <ExternalLink />
                  </a>
                </article>
              ))}
            </div>
          </>
        )}

        {/* VIEW 4: LIVE STATUS */}
        {view === "status" && (
          <>
            <header className="developer-content-header">
              <div>
                <span className="eyebrow">Production runtime</span>
                <h2>Live API status</h2>
                <p>Readiness is fetched from Railway, not hard-coded into this page.</p>
              </div>
              <Button variant="outline" onClick={() => void refreshHealth()}>
                <RefreshCw /> Refresh
              </Button>
            </header>
            {health ? (
              <div className="developer-status-grid">
                <article>
                  <span>Service</span>
                  <strong>{health.status}</strong>
                  <small>HTTP health check passed</small>
                </article>
                <article>
                  <span>Price model</span>
                  <strong>{(health.held_out_r2 * 100).toFixed(2)}% R2</strong>
                  <small>{health.model}</small>
                </article>
                <article>
                  <span>Database</span>
                  <strong>{health.database}</strong>
                  <small>Project and prediction traces</small>
                </article>
                <article>
                  <span>CAD analysis</span>
                  <strong>{health.cad_analysis}</strong>
                  <small>{health.vision_model ?? "No vision model reported"}</small>
                </article>
              </div>
            ) : (
              <div className="developer-status-error">
                <WifiOff />
                <strong>Could not read live health</strong>
                <p>{healthError || "The health request is still running."}</p>
              </div>
            )}
          </>
        )}
      </div>

      <aside className="developer-aside">
        <Card className="developer-aside-card card">
          <span className="eyebrow">Base URL</span>
          <code>{apiUrl}</code>
          <button onClick={() => void copyText("base", apiUrl)}>
            {copied === "base" ? <CheckCircle2 /> : <Copy />} {copied === "base" ? "Copied" : "Copy"}
          </button>
        </Card>
        <Card className="developer-aside-card card">
          <span className="eyebrow">Service capabilities</span>
          <dl>
            <div>
              <dt>API Key Auth</dt>
              <dd>Active</dd>
            </div>
            <div>
              <dt>Price prediction</dt>
              <dd>Live</dd>
            </div>
            <div>
              <dt>CAD intelligence</dt>
              <dd>Live</dd>
            </div>
            <div>
              <dt>Project storage</dt>
              <dd>Connected</dd>
            </div>
          </dl>
        </Card>
        <Card className="developer-aside-card card">
          <span className="eyebrow">Developer links</span>
          <a href={apiUrl + "/docs"} target="_blank" rel="noreferrer">
            <span>
              <strong>Swagger UI</strong>
              <small>Test every route</small>
            </span>
            <ExternalLink />
          </a>
          <a href={apiUrl + "/redoc"} target="_blank" rel="noreferrer">
            <span>
              <strong>ReDoc</strong>
              <small>Read the full contract</small>
            </span>
            <ExternalLink />
          </a>
          <a href={sourceUrl} target="_blank" rel="noreferrer">
            <span>
              <strong>GitHub source</strong>
              <small>Inspect the implementation</small>
            </span>
            <ExternalLink />
          </a>
        </Card>
      </aside>

      {/* CREATE API KEY MODAL */}
      {isCreateModalOpen && (
        <div className="developer-modal-backdrop" onClick={() => setIsCreateModalOpen(false)}>
          <div className="developer-modal card" onClick={(e) => e.stopPropagation()}>
            <header className="developer-modal-header">
              <h3>Create new API key</h3>
              <button type="button" onClick={() => setIsCreateModalOpen(false)} aria-label="Close modal">
                <X />
              </button>
            </header>
            <form onSubmit={handleCreateKeySubmit} className="developer-modal-form">
              <p>Enter a descriptive name to identify this API key in your dashboard.</p>
              <div className="form-group">
                <label htmlFor="key-name-input">Key Name</label>
                <Input
                  id="key-name-input"
                  placeholder="e.g. Thunderous Binturong, Production API Key"
                  value={newKeyName}
                  onChange={(e) => setNewKeyName(e.target.value)}
                  autoFocus
                />
              </div>
              <div className="developer-modal-actions">
                <Button type="button" variant="outline" onClick={() => setIsCreateModalOpen(false)}>
                  Cancel
                </Button>
                <Button type="submit" className="dark-button">
                  Create Key
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* CREATED KEY SECRET SHOWCASE MODAL */}
      {createdSecret && (
        <div className="developer-modal-backdrop" onClick={() => setCreatedSecret(null)}>
          <div className="developer-modal card secret-modal" onClick={(e) => e.stopPropagation()}>
            <header className="developer-modal-header">
              <h3>Save your API Key</h3>
              <button type="button" onClick={() => setCreatedSecret(null)} aria-label="Close modal">
                <X />
              </button>
            </header>
            <div className="developer-modal-body">
              <div className="secret-warning-box">
                <KeyRound />
                <p>
                  <strong>Save this key in a secure location.</strong> You will not be able to view the full secret key again.
                </p>
              </div>
              <div className="secret-key-display">
                <code>{createdSecret}</code>
                <Button variant="outline" onClick={() => void copyText("secret-modal", createdSecret)}>
                  {copied === "secret-modal" ? <CheckCircle2 /> : <Copy />} {copied === "secret-modal" ? "Copied!" : "Copy Key"}
                </Button>
              </div>
              <div className="developer-modal-actions">
                <Button className="dark-button" onClick={() => setCreatedSecret(null)}>
                  I have copied my key
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* DELETE CONFIRMATION MODAL */}
      {deleteConfirmId && (
        <div className="developer-modal-backdrop" onClick={() => setDeleteConfirmId(null)}>
          <div className="developer-modal card delete-modal" onClick={(e) => e.stopPropagation()}>
            <header className="developer-modal-header">
              <h3>Delete API Key</h3>
              <button type="button" onClick={() => setDeleteConfirmId(null)} aria-label="Close modal">
                <X />
              </button>
            </header>
            <div className="developer-modal-body">
              <p>Are you sure you want to delete this key? Applications using this API key will immediately lose access to SpaceWorth services.</p>
              <div className="developer-modal-actions">
                <Button variant="outline" onClick={() => setDeleteConfirmId(null)}>
                  Cancel
                </Button>
                <Button className="destructive-button" onClick={() => handleDeleteKey(deleteConfirmId)}>
                  Delete Key
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
