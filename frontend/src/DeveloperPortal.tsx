import { useEffect, useMemo, useState } from "react";
import {
  Box,
  BookOpen,
  Braces,
  CheckCircle2,
  CircleDot,
  Code2,
  Copy,
  Database,
  Download,
  ExternalLink,
  FileImage,
  Gauge,
  KeyRound,
  RefreshCw,
  Send,
  WifiOff,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

type DeveloperView = "quickstart" | "reference" | "schema" | "status";
type CodeLanguage = "curl" | "javascript" | "python";

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

export function DeveloperPortal({ apiUrl, sourceUrl }: DeveloperPortalProps) {
  const [view, setView] = useState<DeveloperView>("quickstart");
  const [language, setLanguage] = useState<CodeLanguage>("curl");
  const [requestBody, setRequestBody] = useState(() => JSON.stringify(SAMPLE_REQUEST, null, 2));
  const [requestState, setRequestState] = useState<DeveloperRequestState>({ status: "idle" });
  const [health, setHealth] = useState<ApiHealthDetails | null>(null);
  const [healthError, setHealthError] = useState("");
  const [copied, setCopied] = useState("");

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

  const downloadOpenApi = async () => {
    const response = await fetch(apiUrl + "/openapi.json", { cache: "no-store" });
    if (!response.ok) return;
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "spaceworth-openapi.json";
    anchor.click();
    URL.revokeObjectURL(url);
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
          <button className={view === "quickstart" ? "active" : ""} onClick={() => setView("quickstart")}><Send /><span><strong>Quickstart</strong><small>Make a live request</small></span></button>
          <button className={view === "reference" ? "active" : ""} onClick={() => setView("reference")}><BookOpen /><span><strong>API reference</strong><small>Endpoints and methods</small></span></button>
          <button className={view === "schema" ? "active" : ""} onClick={() => setView("schema")}><Braces /><span><strong>OpenAPI schema</strong><small>Export the contract</small></span></button>
          <button className={view === "status" ? "active" : ""} onClick={() => setView("status")}><Gauge /><span><strong>Live status</strong><small>Runtime readiness</small></span></button>
        </nav>
        <div className="developer-help">
          <strong>Need implementation help?</strong>
          <p>Inspect the interactive docs or the public API source.</p>
          <a href={apiUrl + "/docs"} target="_blank" rel="noreferrer">Open Swagger <ExternalLink /></a>
          <a href={sourceUrl + "/blob/main/deployment/house-price-space/app.py"} target="_blank" rel="noreferrer">View API source <Code2 /></a>
        </div>
      </aside>

      <div className="developer-content card">
        {view === "quickstart" && (
          <>
            <header className="developer-content-header">
              <div><span className="eyebrow">Interactive request builder</span><h2>Predict a property price</h2><p>Edit the JSON, send it to production, and copy the same request into your application.</p></div>
              <span className={"developer-live-badge " + (health ? "online" : "")}><i />{health ? "API online" : "Checking API"}</span>
            </header>
            <div className="developer-notice"><KeyRound /><p><strong>No key is exposed in this browser.</strong> The current public project API accepts direct requests. If server authentication is enabled later, keep <code>X-API-Key</code> in your backend or serverless function, never client code.</p></div>
            <section className="developer-endpoint"><span className="method post">POST</span><code>{apiUrl}/predict</code><button onClick={() => void copyText("url", apiUrl + "/predict")} aria-label="Copy endpoint URL">{copied === "url" ? <CheckCircle2 /> : <Copy />}</button></section>
            <div className="developer-builder">
              <section>
                <div className="developer-section-title"><div><span>Request body</span><small>application/json</small></div><button onClick={() => setRequestBody(JSON.stringify(SAMPLE_REQUEST, null, 2))}><RefreshCw /> Reset</button></div>
                <textarea aria-label="Prediction request JSON" spellCheck={false} value={requestBody} onChange={(event) => setRequestBody(event.target.value)} />
              </section>
              <section>
                <div className="developer-section-title"><div><span>Code example</span><small>Ready to paste</small></div><button onClick={() => void copyText("code", generatedCode)}>{copied === "code" ? <CheckCircle2 /> : <Copy />} {copied === "code" ? "Copied" : "Copy"}</button></div>
                <div className="developer-language-tabs">{(["curl", "javascript", "python"] as CodeLanguage[]).map((item) => <button key={item} className={language === item ? "active" : ""} onClick={() => setLanguage(item)}>{item === "javascript" ? "JavaScript" : readable(item)}</button>)}</div>
                <pre><code>{generatedCode}</code></pre>
              </section>
            </div>
            <section className="developer-response">
              <div className="developer-section-title"><div><span>Live response</span><small>{requestState.status === "success" ? requestState.statusCode + " OK · " + requestState.durationMs + " ms" : "Production Railway API"}</small></div><Button className="dark-button" onClick={() => void sendDeveloperRequest()} disabled={requestState.status === "loading"}><Send />{requestState.status === "loading" ? "Sending…" : "Send request"}</Button></div>
              {requestState.status === "success" ? <pre><code>{JSON.stringify(requestState.body, null, 2)}</code></pre> : requestState.status === "error" ? <p className="developer-request-error">{requestState.message}</p> : <div className="developer-response-empty"><CircleDot /><span>Your server response will appear here.</span></div>}
            </section>
          </>
        )}

        {view === "reference" && (
          <>
            <header className="developer-content-header"><div><span className="eyebrow">Version 2.0.0</span><h2>API reference</h2><p>These routes come directly from the live FastAPI OpenAPI document.</p></div><a className="developer-header-link" href={apiUrl + "/redoc"} target="_blank" rel="noreferrer">Open ReDoc <ExternalLink /></a></header>
            <div className="developer-endpoint-list">{endpointRows.map((endpoint) => <article key={endpoint.method + "-" + endpoint.path}><span className={"method " + endpoint.method.toLowerCase()}>{endpoint.method}</span><div><code>{endpoint.path}</code><p>{endpoint.summary}</p></div><a href={apiUrl + "/docs"} target="_blank" rel="noreferrer" aria-label={"Open " + endpoint.path + " in Swagger"}><ExternalLink /></a></article>)}</div>
          </>
        )}

        {view === "schema" && (
          <>
            <header className="developer-content-header"><div><span className="eyebrow">Machine-readable contract</span><h2>Export the SpaceWorth API</h2><p>Download the live schema for SDK generation, API clients, validation, or agent tooling.</p></div><Button className="dark-button" onClick={() => void downloadOpenApi()}><Download /> Download JSON</Button></header>
            <div className="developer-schema-grid"><article><Braces /><span><strong>OpenAPI 3.1</strong><small>Live generated schema</small></span></article><article><Box /><span><strong>PropertyInput</strong><small>Validated prediction payload</small></span></article><article><FileImage /><span><strong>Multipart analysis</strong><small>PNG, JPG, WEBP, or PDF</small></span></article><article><Database /><span><strong>Project traces</strong><small>PostgreSQL-backed persistence</small></span></article></div>
            <section className="developer-schema-preview"><div className="developer-section-title"><div><span>Schema endpoint</span><small>Always reflects the deployed API</small></div><button onClick={() => void copyText("schema", apiUrl + "/openapi.json")}>{copied === "schema" ? <CheckCircle2 /> : <Copy />} Copy URL</button></div><code>{apiUrl}/openapi.json</code><div className="developer-doc-actions"><a href={apiUrl + "/openapi.json"} target="_blank" rel="noreferrer">View raw JSON <ExternalLink /></a><a href={apiUrl + "/docs"} target="_blank" rel="noreferrer">Swagger UI <ExternalLink /></a><a href={apiUrl + "/redoc"} target="_blank" rel="noreferrer">ReDoc <ExternalLink /></a></div></section>
          </>
        )}

        {view === "status" && (
          <>
            <header className="developer-content-header"><div><span className="eyebrow">Production runtime</span><h2>Live API status</h2><p>Readiness is fetched from Railway, not hard-coded into this page.</p></div><Button variant="outline" onClick={() => void refreshHealth()}><RefreshCw /> Refresh</Button></header>
            {health ? <div className="developer-status-grid"><article><span>Service</span><strong>{health.status}</strong><small>HTTP health check passed</small></article><article><span>Price model</span><strong>{(health.held_out_r2 * 100).toFixed(2)}% R2</strong><small>{health.model}</small></article><article><span>Database</span><strong>{health.database}</strong><small>Project and prediction traces</small></article><article><span>CAD analysis</span><strong>{health.cad_analysis}</strong><small>{health.vision_model ?? "No vision model reported"}</small></article></div> : <div className="developer-status-error"><WifiOff /><strong>Could not read live health</strong><p>{healthError || "The health request is still running."}</p></div>}
          </>
        )}
      </div>

      <aside className="developer-aside">
        <Card className="developer-aside-card card"><span className="eyebrow">Base URL</span><code>{apiUrl}</code><button onClick={() => void copyText("base", apiUrl)}>{copied === "base" ? <CheckCircle2 /> : <Copy />} {copied === "base" ? "Copied" : "Copy"}</button></Card>
        <Card className="developer-aside-card card"><span className="eyebrow">Service capabilities</span><dl><div><dt>Price prediction</dt><dd>Live</dd></div><div><dt>CAD intelligence</dt><dd>Live</dd></div><div><dt>Project storage</dt><dd>Connected</dd></div><div><dt>OpenAPI</dt><dd>Exportable</dd></div></dl></Card>
        <Card className="developer-aside-card card"><span className="eyebrow">Developer links</span><a href={apiUrl + "/docs"} target="_blank" rel="noreferrer"><span><strong>Swagger UI</strong><small>Test every route</small></span><ExternalLink /></a><a href={apiUrl + "/redoc"} target="_blank" rel="noreferrer"><span><strong>ReDoc</strong><small>Read the full contract</small></span><ExternalLink /></a><a href={sourceUrl} target="_blank" rel="noreferrer"><span><strong>GitHub source</strong><small>Inspect the implementation</small></span><ExternalLink /></a></Card>
      </aside>
    </section>
  );
}
