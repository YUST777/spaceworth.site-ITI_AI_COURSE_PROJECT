import { useEffect, useMemo, useRef, useState } from "react";
import { Circle, Group, Layer, Line, Rect, Stage, Text } from "react-konva";
import {
  BarChart3,
  Box,
  CheckCircle2,
  DoorOpen,
  Download,
  Expand,
  Home,
  Layers3,
  LayoutGrid,
  List,
  MapPin,
  MousePointer2,
  Plus,
  Save,
  Settings,
  Sparkles,
  Trash2,
  TreePine,
  Type,
  Wifi,
  WifiOff,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

type PropertyForm = {
  title: string;
  description: string;
  areaSqft: string;
  location: string;
  locality: string;
  society: string;
  bedrooms: string;
  bathrooms: string;
  floorNumber: string;
  totalFloors: string;
  furnishing: "unfurnished" | "semi_furnished" | "furnished" | "unknown";
};

type PlanRoom = {
  id: string;
  label: string;
  detail: string;
  x: number;
  y: number;
  width: number;
  height: number;
  accent?: boolean;
  hasDoor?: boolean;
  hasPlant?: boolean;
};

type PredictionState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; price: number }
  | { status: "error"; message: string };

type Unit = {
  id: number;
  name: string;
  form: PropertyForm;
  rooms: PlanRoom[];
  prediction: PredictionState;
};

type Section = "home" | "units" | "plans" | "insights" | "settings";
type CanvasMode = "2d" | "3d";
type Tool = "select" | "door" | "room" | "label" | "plant" | "delete";

const API_URL = (import.meta.env.VITE_PREDICTION_API_URL ?? "").replace(/\/$/, "");
const STORAGE_KEY = "spacemap-project-v1";

const initialForm: PropertyForm = {
  title: "Modern family home",
  description: "A practical home layout with bright living spaces and flexible rooms.",
  areaSqft: "1200",
  location: "thane",
  locality: "kolshet_road",
  society: "lodha_amara",
  bedrooms: "2",
  bathrooms: "2",
  floorNumber: "8",
  totalFloors: "24",
  furnishing: "semi_furnished",
};

const formatCurrency = (value: number) =>
  new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);

function buildPlan(bedrooms: number, areaSqft: number): PlanRoom[] {
  const count = Math.min(4, Math.max(1, Math.round(bedrooms || 2)));
  const areaM2 = Math.max(25, Math.round(areaSqft * 0.092903));
  const bedroomArea = Math.max(9, Math.round(areaM2 * 0.16));
  const templates: PlanRoom[] = [
    { id: "bedroom-1", label: "Bedroom 1", detail: `${bedroomArea} m²`, x: 102, y: 68, width: 178, height: 145, hasDoor: true },
    { id: "bedroom-2", label: "Bedroom 2", detail: `${Math.max(9, bedroomArea - 1)} m²`, x: 348, y: 68, width: 178, height: 145, hasDoor: true },
    { id: "bedroom-3", label: "Bedroom 3", detail: `${Math.max(8, bedroomArea - 2)} m²`, x: 102, y: 226, width: 150, height: 115, hasDoor: true },
    { id: "bedroom-4", label: "Bedroom 4", detail: `${Math.max(8, bedroomArea - 2)} m²`, x: 376, y: 226, width: 150, height: 115, hasDoor: true },
  ];
  const rooms = templates.slice(0, count);
  rooms.push(
    { id: "bathroom-1", label: "Bathroom", detail: "4.0 m²", x: 280, y: 68, width: 68, height: 145, hasDoor: true },
    { id: "living", label: "Living room", detail: `${Math.max(18, Math.round(areaM2 * 0.24))} m²`, x: 102, y: 354, width: 238, height: 202, accent: true, hasDoor: true, hasPlant: true },
    { id: "kitchen", label: "Kitchen & dining", detail: `${Math.max(14, Math.round(areaM2 * 0.18))} m²`, x: 340, y: 354, width: 186, height: 202, accent: true },
    count <= 2
      ? { id: "bathroom-2", label: "Bathroom", detail: "4.0 m²", x: 348, y: 226, width: 178, height: 115, hasDoor: true }
      : { id: "bathroom-2", label: "Bathroom", detail: "4.0 m²", x: 262, y: 226, width: 102, height: 115, hasDoor: true },
  );
  return rooms;
}

function makeInitialUnit(): Unit {
  return {
    id: 1,
    name: "Unit 1",
    form: initialForm,
    rooms: buildPlan(2, 1200),
    prediction: { status: "idle" },
  };
}

function loadUnits(): Unit[] {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) return [makeInitialUnit()];
    const parsed = JSON.parse(saved) as Unit[];
    return Array.isArray(parsed) && parsed.length ? parsed : [makeInitialUnit()];
  } catch {
    return [makeInitialUnit()];
  }
}

function useElementSize<T extends HTMLElement>() {
  const elementRef = useRef<T>(null);
  const [size, setSize] = useState({ width: 680, height: 570 });
  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      setSize({ width: Math.max(280, entry.contentRect.width), height: Math.max(340, entry.contentRect.height) });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [elementRef, size] as const;
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="field">
      <Label>{label}</Label>
      {children}
    </label>
  );
}

function ToolButton({ label, active, destructive, onClick, children }: { label: string; active?: boolean; destructive?: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <Tooltip>
      <TooltipTrigger render={<Button variant="ghost" size="icon" className={`tool ${active ? "selected" : ""} ${destructive ? "danger" : ""}`} onClick={onClick} aria-label={label} />}>
        {children}
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
}

function MiniPlan({ rooms }: { rooms: PlanRoom[] }) {
  return (
    <svg aria-hidden="true" className="mini-plan" viewBox="0 0 90 90">
      <rect x="7" y="6" width="76" height="77" fill="#fff" stroke="#202124" strokeWidth="3" />
      {rooms.slice(0, 7).map((room) => (
        <rect key={room.id} x={8 + ((room.x - 78) / 474) * 74} y={7 + ((room.y - 52) / 520) * 75} width={Math.max(5, (room.width / 474) * 74)} height={Math.max(5, (room.height / 520) * 75)} fill="#f2f3f4" stroke="#555" strokeWidth="1" />
      ))}
    </svg>
  );
}

function ThreeDimensionalPlan({ rooms }: { rooms: PlanRoom[] }) {
  return (
    <div className="three-scene">
      <div className="three-plan">
        {rooms.map((room) => (
          <div
            className={`three-room ${room.accent ? "accent" : ""}`}
            key={room.id}
            style={{
              left: `${((room.x - 78) / 474) * 100}%`,
              top: `${((room.y - 52) / 520) * 100}%`,
              width: `${(room.width / 474) * 100}%`,
              height: `${(room.height / 520) * 100}%`,
            }}
          >
            <span>{room.label}</span>
          </div>
        ))}
      </div>
      <p>Live extrusion of the current 2D room geometry</p>
    </div>
  );
}

function FloorPlan({ rooms, updateRooms, selectedRoom, setSelectedRoom, mode, zoom }: { rooms: PlanRoom[]; updateRooms: (rooms: PlanRoom[]) => void; selectedRoom: string; setSelectedRoom: (id: string) => void; mode: CanvasMode; zoom: number }) {
  const [canvasRef, canvasSize] = useElementSize<HTMLDivElement>();
  const scale = Math.min(1, (canvasSize.width - 26) / 640, (canvasSize.height - 24) / 600) * zoom;
  return (
    <div className="drawing-area" ref={canvasRef}>
      {mode === "3d" ? (
        <ThreeDimensionalPlan rooms={rooms} />
      ) : (
        <Stage width={canvasSize.width} height={canvasSize.height}>
          <Layer x={Math.max(8, (canvasSize.width - 640 * scale) / 2)} y={8} scaleX={scale} scaleY={scale}>
            <Line points={[78, 35, 552, 35]} stroke="#8f949b" strokeWidth={1} />
            <Line points={[78, 29, 78, 41]} stroke="#8f949b" strokeWidth={1} />
            <Line points={[552, 29, 552, 41]} stroke="#8f949b" strokeWidth={1} />
            <Text text="Editable floor plan" x={270} y={17} fontSize={12} fill="#686c73" />
            <Rect x={78} y={52} width={474} height={520} fill="#fff" stroke="#202328" strokeWidth={7} />
            {rooms.map((room) => (
              <Group key={room.id} x={room.x} y={room.y} draggable onClick={() => setSelectedRoom(room.id)} onTap={() => setSelectedRoom(room.id)} onDragEnd={(event) => updateRooms(rooms.map((current) => current.id === room.id ? { ...current, x: event.target.x(), y: event.target.y() } : current))}>
                <Rect width={room.width} height={room.height} fill={room.accent ? "#fafafa" : "#fff"} stroke={selectedRoom === room.id ? "#2563eb" : "#25282d"} strokeWidth={selectedRoom === room.id ? 3 : 4} />
                {room.id.startsWith("bedroom") && <Rect x={18} y={18} width={Math.min(88, room.width - 36)} height={34} fill="#e5e7eb" stroke="#b0b5bd" />}
                {room.id === "living" && <><Rect x={54} y={80} width={92} height={48} cornerRadius={5} fill="#e8eaed" stroke="#b1b6bd" /><Circle x={31} y={105} radius={17} fill="#f0f1f2" stroke="#b1b6bd" /></>}
                {room.id === "kitchen" && <><Rect x={18} y={18} width={room.width - 36} height={22} fill="#e4e6e8" /><Rect x={room.width - 43} y={63} width={22} height={48} fill="#e4e6e8" /></>}
                {room.hasPlant && <><Circle x={room.width - 25} y={room.height - 26} radius={13} fill="#eef3eb" stroke="#778b71" /><Text text="✦" x={room.width - 31} y={room.height - 33} fontSize={14} fill="#60745b" /></>}
                {room.hasDoor && <Line points={[room.width / 2 - 13, room.height, room.width / 2, room.height - 15, room.width / 2 + 13, room.height]} stroke="#555" strokeWidth={2} tension={0.5} />}
                <Text text={room.label} width={room.width} align="center" y={room.height / 2 - 14} fontSize={13} fontStyle="bold" fill="#1d2025" />
                <Text text={room.detail} width={room.width} align="center" y={room.height / 2 + 5} fontSize={11} fill="#575c65" />
              </Group>
            ))}
          </Layer>
        </Stage>
      )}
    </div>
  );
}

function App() {
  const [units, setUnits] = useState<Unit[]>(loadUnits);
  const [activeUnitId, setActiveUnitId] = useState(units[0].id);
  const [activeTab, setActiveTab] = useState<"basic" | "more">("basic");
  const [section, setSection] = useState<Section>("plans");
  const [canvasMode, setCanvasMode] = useState<CanvasMode>("2d");
  const [activeTool, setActiveTool] = useState<Tool>("select");
  const [selectedRoom, setSelectedRoom] = useState("living");
  const [zoom, setZoom] = useState(1);
  const [notice, setNotice] = useState("Ready");
  const [apiHealth, setApiHealth] = useState<"idle" | "checking" | "online" | "offline">("idle");

  const unit = units.find((candidate) => candidate.id === activeUnitId) ?? units[0];
  const form = unit.form;
  const rooms = unit.rooms;
  const prediction = unit.prediction;
  const areaSqft = Number(form.areaSqft) || 0;
  const areaM2 = Math.round(areaSqft * 0.092903);
  const pricePerSqft = prediction.status === "success" && areaSqft > 0 ? prediction.price / areaSqft : null;

  const updateUnit = (updater: (current: Unit) => Unit) => setUnits((current) => current.map((candidate) => candidate.id === activeUnitId ? updater(candidate) : candidate));
  const updateForm = <K extends keyof PropertyForm>(key: K, value: PropertyForm[K]) => updateUnit((current) => ({ ...current, form: { ...current.form, [key]: value }, prediction: { status: "idle" } }));
  const updateRooms = (nextRooms: PlanRoom[]) => updateUnit((current) => ({ ...current, rooms: nextRooms }));

  const generatePlan = () => {
    const next = buildPlan(Number(form.bedrooms), areaSqft);
    updateRooms(next);
    setSelectedRoom(next.find((room) => room.id === "living")?.id ?? next[0]?.id ?? "");
    setSection("plans");
    setNotice(`Generated ${form.bedrooms || 1}-bedroom plan from the current inputs`);
  };

  const predict = async () => {
    generatePlan();
    if (!API_URL) {
      updateUnit((current) => ({ ...current, prediction: { status: "error", message: "Real model API is not configured yet. No fake price was generated." } }));
      setNotice("Prediction stopped: real model API is not connected");
      return;
    }
    updateUnit((current) => ({ ...current, prediction: { status: "loading" } }));
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          area_sqft: areaSqft,
          area_type: "super",
          location: form.location,
          locality: form.locality || undefined,
          society: form.society || undefined,
          bedrooms: Number(form.bedrooms) || undefined,
          bathroom: Number(form.bathrooms) || undefined,
          floor_num: Number(form.floorNumber) || undefined,
          total_floors: Number(form.totalFloors) || undefined,
          furnishing: form.furnishing,
        }),
      });
      const result = (await response.json()) as { predicted_price_inr?: number; detail?: string };
      if (!response.ok || typeof result.predicted_price_inr !== "number") throw new Error(result.detail ?? "Invalid prediction response");
      updateUnit((current) => ({ ...current, prediction: { status: "success", price: result.predicted_price_inr! } }));
      setNotice("Real model prediction received");
    } catch (error) {
      updateUnit((current) => ({ ...current, prediction: { status: "error", message: error instanceof Error ? error.message : "Prediction request failed" } }));
      setNotice("Prediction request failed");
    }
  };

  const applyTool = (tool: Tool) => {
    setActiveTool(tool);
    if (tool === "select") return;
    if (tool === "room") {
      const nextIndex = rooms.filter((room) => room.id.startsWith("custom-room")).length + 1;
      const newRoom: PlanRoom = { id: `custom-room-${Date.now()}`, label: `Flex room ${nextIndex}`, detail: "10 m²", x: 220 + nextIndex * 9, y: 260 + nextIndex * 8, width: 125, height: 92, hasDoor: true };
      updateRooms([...rooms, newRoom]);
      setSelectedRoom(newRoom.id);
      setNotice(`${newRoom.label} added; drag it into position`);
      return;
    }
    const current = rooms.find((room) => room.id === selectedRoom);
    if (!current) {
      setNotice("Select a room first");
      return;
    }
    if (tool === "delete") {
      updateRooms(rooms.filter((room) => room.id !== selectedRoom));
      setSelectedRoom(rooms.find((room) => room.id !== selectedRoom)?.id ?? "");
      setNotice(`${current.label} removed`);
    } else if (tool === "door") {
      updateRooms(rooms.map((room) => room.id === selectedRoom ? { ...room, hasDoor: !room.hasDoor } : room));
      setNotice(`Door ${current.hasDoor ? "removed from" : "added to"} ${current.label}`);
    } else if (tool === "plant") {
      updateRooms(rooms.map((room) => room.id === selectedRoom ? { ...room, hasPlant: !room.hasPlant } : room));
      setNotice(`Plant ${current.hasPlant ? "removed from" : "added to"} ${current.label}`);
    } else if (tool === "label") {
      const label = current.label.startsWith("Custom") ? "Flex space" : `Custom ${current.label}`;
      updateRooms(rooms.map((room) => room.id === selectedRoom ? { ...room, label } : room));
      setNotice(`${current.label} label updated`);
    }
    setActiveTool("select");
  };

  const addUnit = () => {
    const id = Math.max(0, ...units.map((candidate) => candidate.id)) + 1;
    const next: Unit = { id, name: `Unit ${id}`, form: { ...initialForm, title: `New property ${id}` }, rooms: buildPlan(2, 1200), prediction: { status: "idle" } };
    setUnits((current) => [...current, next]);
    setActiveUnitId(id);
    setSection("plans");
    setNotice(`${next.name} created`);
  };

  const removeUnit = (id: number) => {
    if (units.length === 1) {
      setNotice("A project must keep at least one unit");
      return;
    }
    const next = units.filter((candidate) => candidate.id !== id);
    setUnits(next);
    if (activeUnitId === id) setActiveUnitId(next[0].id);
    setNotice(`Unit ${id} removed`);
  };

  const saveProject = () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(units));
    setNotice("Project saved in this browser");
  };

  const exportProject = () => {
    const blob = new Blob([JSON.stringify({ version: 1, exportedAt: new Date().toISOString(), units }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "spacemap-project.json";
    link.click();
    URL.revokeObjectURL(url);
    setNotice("Project JSON exported");
  };

  const checkApi = async () => {
    if (!API_URL) {
      setApiHealth("offline");
      setNotice("No API URL configured");
      return;
    }
    setApiHealth("checking");
    try {
      const response = await fetch(`${API_URL}/health`);
      setApiHealth(response.ok ? "online" : "offline");
      setNotice(response.ok ? "Model API is online" : "Model API health check failed");
    } catch {
      setApiHealth("offline");
      setNotice("Model API is unreachable");
    }
  };

  const openSection = (next: Section) => {
    setSection(next);
    if (next === "home" || next === "plans") setCanvasMode("2d");
  };

  const centerPanel = useMemo(() => {
    if (section === "units") {
      return <div className="functional-panel"><div className="functional-heading"><div><h2>Property units</h2><p>Every card is backed by editable project state.</p></div><Button onClick={addUnit}><Plus /> Add unit</Button></div><div className="unit-grid">{units.map((candidate) => <button key={candidate.id} className={`unit-grid-card ${candidate.id === activeUnitId ? "active" : ""}`} onClick={() => { setActiveUnitId(candidate.id); setSection("plans"); }}><MiniPlan rooms={candidate.rooms} /><span><strong>{candidate.name}</strong><small>{candidate.form.areaSqft} sq ft · {candidate.form.bedrooms} bedrooms</small></span>{candidate.prediction.status === "success" && <em>{formatCurrency(candidate.prediction.price)}</em>}</button>)}</div></div>;
    }
    if (section === "insights") {
      return <div className="functional-panel"><div className="functional-heading"><div><h2>Live project insights</h2><p>Calculated from your current units; no sample analytics.</p></div></div><div className="insight-grid"><article><span>Total units</span><strong>{units.length}</strong></article><article><span>Combined area</span><strong>{units.reduce((sum, candidate) => sum + (Number(candidate.form.areaSqft) || 0), 0).toLocaleString("en-IN")} sq ft</strong></article><article><span>Total rooms</span><strong>{units.reduce((sum, candidate) => sum + candidate.rooms.length, 0)}</strong></article><article><span>Real predictions</span><strong>{units.filter((candidate) => candidate.prediction.status === "success").length}</strong></article></div><div className="insight-list">{units.map((candidate) => <div key={candidate.id}><span>{candidate.name}</span><b>{candidate.form.location || "No location"}</b><em>{candidate.prediction.status === "success" ? formatCurrency(candidate.prediction.price) : "Not predicted"}</em></div>)}</div></div>;
    }
    if (section === "settings") {
      return <div className="functional-panel"><div className="functional-heading"><div><h2>Workspace settings</h2><p>Connection and local project controls.</p></div></div><div className="settings-stack"><article><div className="api-state">{apiHealth === "online" ? <Wifi /> : <WifiOff />}<div><strong>Prediction API</strong><span>{API_URL || "Not configured"}</span></div></div><Button variant="outline" onClick={checkApi}>{apiHealth === "checking" ? "Checking…" : "Check connection"}</Button></article><article><div><strong>Browser project storage</strong><span>Save keeps the editable project on this device.</span></div><Button variant="outline" onClick={saveProject}>Save now</Button></article><article><div><strong>Portable project file</strong><span>Export all forms, plans and real predictions as JSON.</span></div><Button variant="outline" onClick={exportProject}>Export JSON</Button></article></div></div>;
    }
    return <><div className="canvas-header"><div><h2>Visual floor plan</h2><p>{areaM2 || "—"} m² <span>•</span> Floor {form.floorNumber || "—"} <span>•</span> {rooms.length} spaces</p></div><div className="canvas-modes"><Button className={canvasMode === "2d" ? "dark-button compact" : "soft-button compact"} onClick={() => setCanvasMode("2d")}>2D plan</Button><Button className={canvasMode === "3d" ? "dark-button compact" : "soft-button compact"} onClick={() => setCanvasMode("3d")}>3D view</Button><Button variant="outline" size="icon" className="soft-button icon-button" onClick={() => document.documentElement.requestFullscreen?.()} aria-label="Fullscreen"><Expand /></Button></div></div><FloorPlan rooms={rooms} updateRooms={updateRooms} selectedRoom={selectedRoom} setSelectedRoom={setSelectedRoom} mode={canvasMode} zoom={zoom} /><div className="canvas-toolbar"><ToolButton label="Select and move" active={activeTool === "select"} onClick={() => applyTool("select")}><MousePointer2 /></ToolButton><ToolButton label="Toggle door on selected room" onClick={() => applyTool("door")}><DoorOpen /></ToolButton><ToolButton label="Add a room" onClick={() => applyTool("room")}><LayoutGrid /></ToolButton><ToolButton label="Rename selected room" onClick={() => applyTool("label")}><Type /></ToolButton><ToolButton label="Toggle plant" onClick={() => applyTool("plant")}><TreePine /></ToolButton><Separator orientation="vertical" className="toolbar-divider" /><ToolButton label="Delete selected room" destructive onClick={() => applyTool("delete")}><Trash2 /></ToolButton></div><div className="zoom-controls"><Button variant="ghost" size="icon-xs" onClick={() => setZoom((value) => Math.max(.65, value - .1))}><ZoomOut /></Button><span>{Math.round(zoom * 100)}%</span><Button variant="ghost" size="icon-xs" onClick={() => setZoom((value) => Math.min(1.35, value + .1))}><ZoomIn /></Button></div></>;
  }, [section, units, activeUnitId, areaM2, form, rooms, selectedRoom, canvasMode, zoom, activeTool, apiHealth]);

  return (
    <main className="app-shell">
      <Card className="topbar card">
        <div className="brand"><Box /><span>SpaceMap</span><em>AI</em></div>
        <div className="view-switch" aria-label="Workspace view"><Button variant="ghost" className={section === "units" ? "active" : ""} onClick={() => openSection("units")}><List /> List view</Button><Button variant="ghost" className={section !== "units" ? "active" : ""} onClick={() => openSection("plans")}><LayoutGrid /> Visual view</Button></div>
        <div className="header-actions"><span className="save-status"><CheckCircle2 /> {notice}</span><Button variant="outline" className="soft-button" onClick={saveProject}><Save /> Save</Button><Button className="dark-button" onClick={exportProject}><Download /> Export</Button></div>
      </Card>

      <div className="workspace">
        <nav className="rail card" aria-label="Primary navigation">
          {[{ id: "home", label: "Home", icon: Home }, { id: "units", label: "Units", icon: LayoutGrid }, { id: "plans", label: "Plans", icon: Layers3 }, { id: "insights", label: "Insights", icon: BarChart3 }].map(({ id, label, icon: Icon }) => <Tooltip key={id}><TooltipTrigger render={<Button variant="ghost" size="icon" className={`rail-item ${section === id ? "active" : ""}`} onClick={() => openSection(id as Section)} aria-label={label} />}><Icon /></TooltipTrigger><TooltipContent side="right">{label}</TooltipContent></Tooltip>)}
          <Tooltip><TooltipTrigger render={<Button variant="ghost" size="icon" className={`rail-item push-bottom ${section === "settings" ? "active" : ""}`} onClick={() => openSection("settings")} aria-label="Settings" />}><Settings /></TooltipTrigger><TooltipContent side="right">Settings</TooltipContent></Tooltip>
        </nav>

        <Card className="details-panel card">
          <div className="panel-heading"><h1>Property details</h1><p>Inputs used by the floor plan and real model request.</p></div>
          <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as "basic" | "more")}><TabsList variant="line" className="tabs"><TabsTrigger value="basic">Basic info</TabsTrigger><TabsTrigger value="more">More details</TabsTrigger></TabsList></Tabs>
          {activeTab === "basic" ? <div className="form-stack"><Field label="Title"><Input value={form.title} onChange={(event) => updateForm("title", event.target.value)} /></Field><Field label="Description"><Textarea value={form.description} onChange={(event) => updateForm("description", event.target.value)} /></Field><div className="field-grid"><Field label="Area (sq ft)"><Input type="number" min="100" value={form.areaSqft} onChange={(event) => updateForm("areaSqft", event.target.value)} /></Field><Field label="Bedrooms"><Input type="number" min="1" max="4" value={form.bedrooms} onChange={(event) => updateForm("bedrooms", event.target.value)} /></Field></div><Field label="Location"><span className="input-with-icon"><Input value={form.location} onChange={(event) => updateForm("location", event.target.value)} /><MapPin /></span></Field><div className="field-grid"><Field label="Bathrooms"><Input type="number" min="0" max="20" value={form.bathrooms} onChange={(event) => updateForm("bathrooms", event.target.value)} /></Field><Field label="Furnishing"><Select value={form.furnishing} onValueChange={(value) => updateForm("furnishing", value as PropertyForm["furnishing"])}><SelectTrigger className="w-full"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="unfurnished">Unfurnished</SelectItem><SelectItem value="semi_furnished">Semi furnished</SelectItem><SelectItem value="furnished">Furnished</SelectItem><SelectItem value="unknown">Unknown</SelectItem></SelectContent></Select></Field></div></div> : <div className="form-stack"><Field label="Locality"><Input value={form.locality} onChange={(event) => updateForm("locality", event.target.value)} /></Field><Field label="Society"><Input value={form.society} onChange={(event) => updateForm("society", event.target.value)} /></Field><div className="field-grid"><Field label="Current floor"><Input type="number" min="-2" value={form.floorNumber} onChange={(event) => updateForm("floorNumber", event.target.value)} /></Field><Field label="Total floors"><Input type="number" min="1" value={form.totalFloors} onChange={(event) => updateForm("totalFloors", event.target.value)} /></Field></div><div className="help-card"><Sparkles /><p>Price and price-per-square-foot are never inputs. Prediction stays empty until the real API answers.</p></div></div>}
          <div className="panel-cta"><Button variant="outline" className="outline-button" onClick={generatePlan}>Generate floor plan</Button><Button className="dark-button wide" onClick={predict} disabled={prediction.status === "loading"}>{prediction.status === "loading" ? "Predicting…" : "Generate & predict"}</Button></div>
        </Card>

        <Card className="canvas-panel card">{centerPanel}</Card>

        <aside className="summary-column">
          <Card className="summary-card card"><h2>Property summary</h2><dl><div><dt>Super area</dt><dd>{areaSqft || "—"} sq ft</dd></div><div><dt>Bedrooms</dt><dd>{form.bedrooms || "—"}</dd></div><div><dt>Location</dt><dd>{form.location || "—"}</dd></div><div><dt>Furnishing</dt><dd>{form.furnishing.replace("_", " ")}</dd></div><div><dt>Prediction</dt><dd className={prediction.status === "success" ? "price-result" : ""}>{prediction.status === "success" ? formatCurrency(prediction.price) : prediction.status === "loading" ? "Calculating…" : "Not predicted"}</dd></div>{pricePerSqft && <div><dt>Predicted / sq ft</dt><dd>{formatCurrency(pricePerSqft)}</dd></div>}</dl>{prediction.status === "error" && <p className="api-note">{prediction.message}</p>}<Button variant="outline" className="outline-button full" onClick={() => setActiveTab("more")}>Edit all details</Button></Card>
          <Card className="units-card card"><div className="units-title"><div><h2>Project units</h2><p>{units.length} editable {units.length === 1 ? "unit" : "units"}</p></div><Button size="icon-sm" onClick={addUnit}><Plus /></Button></div><div className="unit-list">{units.map((candidate) => <article className={candidate.id === activeUnitId ? "unit active" : "unit"} key={candidate.id} onClick={() => setActiveUnitId(candidate.id)}><MiniPlan rooms={candidate.rooms} /><div><strong>{candidate.name}</strong><span>{candidate.form.areaSqft} sq ft</span></div><Button variant="ghost" size="icon-xs" aria-label={`Remove ${candidate.name}`} onClick={(event) => { event.stopPropagation(); removeUnit(candidate.id); }}>×</Button></article>)}</div><Button variant="outline" className="add-unit" onClick={addUnit}><Plus /> Add new unit</Button></Card>
        </aside>
      </div>
    </main>
  );
}

export default App;
