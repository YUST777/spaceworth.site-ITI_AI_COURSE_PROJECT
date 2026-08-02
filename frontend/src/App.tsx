import { useEffect, useMemo, useRef, useState } from "react";
import { Circle, Group, Layer, Line, Rect, Stage, Text } from "react-konva";
import {
  Box,
  Database,
  DoorOpen,
  ExternalLink,
  Expand,
  FileImage,
  Grid2X2,
  Layers3,
  Magnet,
  MapPin,
  MapPinned,
  MousePointer2,
  Redo2,
  RotateCcw,
  Ruler,
  Settings,
  Sparkles,
  Trash2,
  TreePine,
  Type,
  Upload,
  Undo2,
  Wifi,
  WifiOff,
  X,
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
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

type PropertyForm = {
  areaSqft: string;
  areaType: "carpet" | "super";
  location: string;
  locality: string;
  society: string;
  bedrooms: string;
  bathrooms: string;
  balcony: string;
  carParking: string;
  floorNumber: string;
  totalFloors: string;
  propertyType: "flat" | "villa" | "house" | "builder_floor" | "penthouse" | "studio" | "plot" | "unknown";
  furnishing: "unfurnished" | "semi_furnished" | "furnished" | "unknown";
  transaction: "resale" | "new_property" | "other" | "unknown";
  ownership: "freehold" | "cooperative_society" | "leasehold" | "unknown";
  facing: string;
  overlooking: string;
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

type ProjectState = {
  form: PropertyForm;
  rooms: PlanRoom[];
  prediction: PredictionState;
};

type UploadedPlan = {
  name: string;
  size: number;
  type: string;
  url: string;
};

type UploadAnalysisState =
  | { status: "idle" }
  | { status: "ready" }
  | { status: "loading" }
  | { status: "success"; price: number }
  | { status: "error"; message: string };

type Section = "plan" | "upload" | "settings";
type CanvasMode = "2d" | "3d";
type Tool = "select" | "door" | "room" | "label" | "plant" | "delete";
type DatabaseSyncState = "loading" | "syncing" | "synced" | "offline";

const API_URL = (
  import.meta.env.VITE_PREDICTION_API_URL ??
  "https://iti-house-price-api-production.up.railway.app"
).replace(/\/$/, "");
const PLAN_API_URL = (import.meta.env.VITE_PLAN_ANALYSIS_API_URL ?? "").replace(/\/$/, "");
const STORAGE_KEY = "spacemap-project-v2";
const PROJECT_ID_KEY = "spacemap-project-id-v1";
const PREDICTION_COOLDOWN_MS = 1800;

const NUMERIC_LIMITS = {
  areaSqft: { min: 100, max: 25000, fallback: 1200 },
  bedrooms: { min: 1, max: 4, fallback: 2 },
  bathrooms: { min: 1, max: 3, fallback: 2 },
  balcony: { min: 0, max: 4, fallback: 1 },
  carParking: { min: 0, max: 4, fallback: 1 },
  floorNumber: { min: -2, max: 250, fallback: 8 },
  totalFloors: { min: 1, max: 250, fallback: 24 },
} as const;

type NumericFormKey = keyof typeof NUMERIC_LIMITS;

function getProjectId() {
  const saved = localStorage.getItem(PROJECT_ID_KEY);
  if (saved) return saved;
  const generated = crypto.randomUUID();
  localStorage.setItem(PROJECT_ID_KEY, generated);
  return generated;
}

const PROJECT_ID = getProjectId();

const initialForm: PropertyForm = {
  areaSqft: "1200",
  areaType: "super",
  location: "thane",
  locality: "kolshet road",
  society: "lodha amara",
  bedrooms: "2",
  bathrooms: "2",
  balcony: "1",
  carParking: "1",
  floorNumber: "8",
  totalFloors: "24",
  propertyType: "flat",
  furnishing: "semi_furnished",
  transaction: "resale",
  ownership: "freehold",
  facing: "east",
  overlooking: "garden",
};

const formatCurrency = (value: number) =>
  new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);

const readable = (value: string) =>
  value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");

const optionalNumber = (value: string) => (value.trim() === "" ? undefined : Number(value));

function normalizeNumericField(key: NumericFormKey, value: string) {
  const limits = NUMERIC_LIMITS[key];
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return String(limits.fallback);
  return String(Math.min(limits.max, Math.max(limits.min, Math.round(parsed))));
}

function normalizeForm(form: PropertyForm): PropertyForm {
  const next = { ...form };
  (Object.keys(NUMERIC_LIMITS) as NumericFormKey[]).forEach((key) => {
    next[key] = normalizeNumericField(key, next[key]);
  });
  const floor = Number(next.floorNumber);
  const totalFloors = Number(next.totalFloors);
  next.floorNumber = String(Math.min(Math.max(-2, floor), totalFloors));
  if (!next.location.trim()) next.location = initialForm.location;
  return next;
}

function buildPlan(bedrooms: number, bathrooms: number, areaSqft: number): PlanRoom[] {
  const bedroomCount = Math.min(4, Math.max(1, Math.round(bedrooms || 2)));
  const bathroomCount = Math.min(3, Math.max(1, Math.round(bathrooms || 1)));
  const areaM2 = Math.max(25, Math.round(areaSqft * 0.092903));
  const bedroomArea = Math.max(9, Math.round(areaM2 * 0.16));
  const bedroomTemplates: PlanRoom[] = [
    { id: "bedroom-1", label: "Bedroom 1", detail: `${bedroomArea} m²`, x: 102, y: 68, width: 178, height: 145, hasDoor: true },
    { id: "bedroom-2", label: "Bedroom 2", detail: `${Math.max(9, bedroomArea - 1)} m²`, x: 348, y: 68, width: 178, height: 145, hasDoor: true },
    { id: "bedroom-3", label: "Bedroom 3", detail: `${Math.max(8, bedroomArea - 2)} m²`, x: 102, y: 226, width: 150, height: 115, hasDoor: true },
    { id: "bedroom-4", label: "Bedroom 4", detail: `${Math.max(8, bedroomArea - 2)} m²`, x: 376, y: 226, width: 150, height: 115, hasDoor: true },
  ];
  const bathroomTemplates: PlanRoom[] = [
    { id: "bathroom-1", label: "Bathroom 1", detail: "4.0 m²", x: 280, y: 68, width: 68, height: 145, hasDoor: true },
    bedroomCount <= 2
      ? { id: "bathroom-2", label: "Bathroom 2", detail: "4.0 m²", x: 348, y: 226, width: 178, height: 115, hasDoor: true }
      : { id: "bathroom-2", label: "Bathroom 2", detail: "4.0 m²", x: 262, y: 226, width: 102, height: 115, hasDoor: true },
    { id: "bathroom-3", label: "Powder room", detail: "2.5 m²", x: 280, y: 226, width: 78, height: 88, hasDoor: true },
  ];

  return [
    ...bedroomTemplates.slice(0, bedroomCount),
    ...bathroomTemplates.slice(0, bathroomCount),
    { id: "living", label: "Living room", detail: `${Math.max(18, Math.round(areaM2 * 0.24))} m²`, x: 102, y: 354, width: 238, height: 202, accent: true, hasDoor: true, hasPlant: true },
    { id: "kitchen", label: "Kitchen & dining", detail: `${Math.max(14, Math.round(areaM2 * 0.18))} m²`, x: 340, y: 354, width: 186, height: 202, accent: true },
  ];
}

function loadProject(): ProjectState {
  const fallback: ProjectState = {
    form: initialForm,
    rooms: buildPlan(2, 2, 1200),
    prediction: { status: "idle" },
  };

  try {
    const saved = localStorage.getItem(STORAGE_KEY) ?? localStorage.getItem("spacemap-project-v1");
    if (!saved) return fallback;
    const parsed = JSON.parse(saved) as ProjectState | Array<{ form?: Partial<PropertyForm>; rooms?: PlanRoom[]; prediction?: PredictionState }>;
    const project = Array.isArray(parsed) ? parsed[0] : parsed;
    if (!project) return fallback;
    return {
      form: normalizeForm({ ...initialForm, ...project.form }),
      rooms: Array.isArray(project.rooms) && project.rooms.length ? project.rooms : fallback.rooms,
      prediction: project.prediction ?? { status: "idle" },
    };
  } catch {
    return fallback;
  }
}

function useElementSize<T extends HTMLElement>() {
  const elementRef = useRef<T>(null);
  const [size, setSize] = useState({ width: 680, height: 570 });

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      setSize({
        width: Math.max(280, entry.contentRect.width),
        height: Math.max(340, entry.contentRect.height),
      });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return [elementRef, size] as const;
}

function Field({ label, children, wide = false }: { label: string; children: React.ReactNode; wide?: boolean }) {
  return (
    <label className={`field ${wide ? "wide-field" : ""}`}>
      <Label>{label}</Label>
      {children}
    </label>
  );
}

function SafeNumberInput({
  field,
  value,
  onChange,
  onBlur,
}: {
  field: NumericFormKey;
  value: string;
  onChange: (value: string) => void;
  onBlur: () => void;
}) {
  const limits = NUMERIC_LIMITS[field];
  return (
    <Input
      type="number"
      inputMode="numeric"
      min={limits.min}
      max={limits.max}
      step="1"
      value={value}
      onChange={(event) => onChange(event.target.value)}
      onBlur={onBlur}
    />
  );
}

function ToolButton({
  label,
  active,
  destructive,
  onClick,
  children,
}: {
  label: string;
  active?: boolean;
  destructive?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <Button
            variant="ghost"
            size="icon"
            className={`tool ${active ? "selected" : ""} ${destructive ? "danger" : ""}`}
            onClick={onClick}
            aria-label={label}
          />
        }
      >
        {children}
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
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
      <p>Live 3D view of the current editable room layout</p>
    </div>
  );
}

function FloorPlan({
  rooms,
  updateRooms,
  selectedRoom,
  setSelectedRoom,
  mode,
  zoom,
  snapToGrid,
  showGrid,
  showDimensions,
}: {
  rooms: PlanRoom[];
  updateRooms: (rooms: PlanRoom[]) => void;
  selectedRoom: string;
  setSelectedRoom: (id: string) => void;
  mode: CanvasMode;
  zoom: number;
  snapToGrid: boolean;
  showGrid: boolean;
  showDimensions: boolean;
}) {
  const [canvasRef, canvasSize] = useElementSize<HTMLDivElement>();
  const scale = Math.min(1, (canvasSize.width - 26) / 640, (canvasSize.height - 24) / 600) * zoom;
  const snap = (value: number) => (snapToGrid ? Math.round(value / 8) * 8 : value);
  const roomFill = (room: PlanRoom) => {
    if (room.id.startsWith("bedroom")) return "#f7f8fa";
    if (room.id.startsWith("bathroom")) return "#f1f5f8";
    if (room.id === "kitchen") return "#f4f5f1";
    if (room.id === "living") return "#f8f7f3";
    return room.accent ? "#f8f9fa" : "#fff";
  };

  return (
    <div className="drawing-area" ref={canvasRef}>
      {mode === "3d" ? (
        <ThreeDimensionalPlan rooms={rooms} />
      ) : (
        <Stage width={canvasSize.width} height={canvasSize.height}>
          <Layer
            x={Math.max(8, (canvasSize.width - 640 * scale) / 2)}
            y={8}
            scaleX={scale}
            scaleY={scale}
          >
            {showGrid && Array.from({ length: 30 }, (_, index) => (
              <Line key={`vertical-grid-${index}`} points={[78 + index * 16, 52, 78 + index * 16, 572]} stroke="#eef0f2" strokeWidth={0.7} />
            ))}
            {showGrid && Array.from({ length: 33 }, (_, index) => (
              <Line key={`horizontal-grid-${index}`} points={[78, 52 + index * 16, 552, 52 + index * 16]} stroke="#eef0f2" strokeWidth={0.7} />
            ))}
            <Line points={[78, 35, 552, 35]} stroke="#8f949b" strokeWidth={1} />
            <Line points={[78, 29, 78, 41]} stroke="#8f949b" strokeWidth={1} />
            <Line points={[552, 29, 552, 41]} stroke="#8f949b" strokeWidth={1} />
            <Text text="Editable floor plan" x={270} y={17} fontSize={12} fill="#686c73" />
            <Rect x={78} y={52} width={474} height={520} fill="#fff" stroke="#202328" strokeWidth={7} />
            {rooms.map((room) => (
              <Group
                key={room.id}
                x={room.x}
                y={room.y}
                draggable
                onClick={() => setSelectedRoom(room.id)}
                onTap={() => setSelectedRoom(room.id)}
                onDragEnd={(event) =>
                  updateRooms(
                    rooms.map((current) =>
                      current.id === room.id
                        ? {
                            ...current,
                            x: Math.min(552 - current.width, Math.max(78, snap(event.target.x()))),
                            y: Math.min(572 - current.height, Math.max(52, snap(event.target.y()))),
                          }
                        : current,
                    ),
                  )
                }
              >
                <Rect
                  width={room.width}
                  height={room.height}
                  fill={roomFill(room)}
                  stroke={selectedRoom === room.id ? "#2563eb" : "#25282d"}
                  strokeWidth={selectedRoom === room.id ? 3 : 4}
                />
                {room.id.startsWith("bedroom") && (
                  <Rect
                    x={18}
                    y={18}
                    width={Math.min(88, room.width - 36)}
                    height={34}
                    fill="#e5e7eb"
                    stroke="#b0b5bd"
                  />
                )}
                {room.id === "living" && (
                  <>
                    <Rect x={54} y={80} width={92} height={48} cornerRadius={5} fill="#e8eaed" stroke="#b1b6bd" />
                    <Circle x={31} y={105} radius={17} fill="#f0f1f2" stroke="#b1b6bd" />
                  </>
                )}
                {room.id === "kitchen" && (
                  <>
                    <Rect x={18} y={18} width={room.width - 36} height={22} fill="#e4e6e8" />
                    <Rect x={room.width - 43} y={63} width={22} height={48} fill="#e4e6e8" />
                  </>
                )}
                {room.hasPlant && (
                  <>
                    <Circle x={room.width - 25} y={room.height - 26} radius={13} fill="#eef3eb" stroke="#778b71" />
                    <Text text="✦" x={room.width - 31} y={room.height - 33} fontSize={14} fill="#60745b" />
                  </>
                )}
                {room.hasDoor && (
                  <Line
                    points={[room.width / 2 - 13, room.height, room.width / 2, room.height - 15, room.width / 2 + 13, room.height]}
                    stroke="#555"
                    strokeWidth={2}
                    tension={0.5}
                  />
                )}
                <Text
                  text={room.width < 90 ? room.label.replace("Bathroom", "Bath") : room.label}
                  width={room.width}
                  align="center"
                  y={room.height / 2 - 14}
                  fontSize={13}
                  fontStyle="bold"
                  fill="#1d2025"
                />
                <Text
                  text={room.detail}
                  width={room.width}
                  align="center"
                  y={room.height / 2 + 5}
                  fontSize={11}
                  fill="#575c65"
                />
                {showDimensions && (
                  <>
                    <Line points={[0, -8, room.width, -8]} stroke="#9aa1aa" strokeWidth={1} />
                    <Text text={`${Math.max(2, Math.round(room.width / 14))} m`} x={0} y={-22} width={room.width} align="center" fontSize={8} fill="#737b84" />
                  </>
                )}
              </Group>
            ))}
          </Layer>
        </Stage>
      )}
    </div>
  );
}

function App() {
  const [project, setProject] = useState<ProjectState>(loadProject);
  const [section, setSection] = useState<Section>("plan");
  const [canvasMode, setCanvasMode] = useState<CanvasMode>("2d");
  const [activeTool, setActiveTool] = useState<Tool>("select");
  const [selectedRoom, setSelectedRoom] = useState("living");
  const [zoom, setZoom] = useState(1);
  const [snapToGrid, setSnapToGrid] = useState(true);
  const [showGrid, setShowGrid] = useState(true);
  const [showDimensions, setShowDimensions] = useState(true);
  const [apiHealth, setApiHealth] = useState<"checking" | "online" | "offline">("checking");
  const [databaseSync, setDatabaseSync] = useState<DatabaseSyncState>("loading");
  const [remoteReady, setRemoteReady] = useState(false);
  const [syncRetryNonce, setSyncRetryNonce] = useState(0);
  const [mapOpen, setMapOpen] = useState(false);
  const [uploadedPlan, setUploadedPlan] = useState<UploadedPlan | null>(null);
  const [uploadAnalysis, setUploadAnalysis] = useState<UploadAnalysisState>({ status: "idle" });
  const [draggingPlan, setDraggingPlan] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const initialPlanCheck = useRef(true);
  const lastPredictionRequestAt = useRef(0);
  const roomHistory = useRef<PlanRoom[][]>([]);
  const roomFuture = useRef<PlanRoom[][]>([]);
  const [historyVersion, setHistoryVersion] = useState(0);

  const { form, rooms, prediction } = project;
  const safeFormView = normalizeForm(form);
  const areaSqft = Number(safeFormView.areaSqft);
  const areaM2 = Math.round(areaSqft * 0.092903);
  const selectedRoomData = rooms.find((room) => room.id === selectedRoom);
  const layoutBedrooms = rooms.filter((room) => room.id.startsWith("bedroom")).length;
  const layoutBathrooms = rooms.filter((room) => room.id.startsWith("bathroom")).length;
  const pricePerSqft = prediction.status === "success" && areaSqft > 0 ? prediction.price / areaSqft : null;
  const mapQuery = [form.locality, form.location, "India"].filter(Boolean).join(", ");
  const mapEmbedUrl = `https://www.google.com/maps?q=${encodeURIComponent(mapQuery)}&output=embed`;
  const mapPageUrl = `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(mapQuery)}`;

  const updateForm = <K extends keyof PropertyForm>(key: K, value: PropertyForm[K]) => {
    setProject((current) => ({
      ...current,
      form: { ...current.form, [key]: value },
      prediction: { status: "idle" },
    }));
  };

  const normalizeCurrentForm = () => {
    setProject((current) => ({
      ...current,
      form: normalizeForm(current.form),
      prediction: { status: "idle" },
    }));
  };

  const updateRooms = (nextRooms: PlanRoom[]) => {
    roomHistory.current = [...roomHistory.current.slice(-29), rooms];
    roomFuture.current = [];
    setProject((current) => ({ ...current, rooms: nextRooms }));
    setHistoryVersion((version) => version + 1);
  };

  const undoRooms = () => {
    const previousRooms = roomHistory.current.pop();
    if (!previousRooms) return;
    roomFuture.current = [...roomFuture.current, rooms];
    setProject((current) => ({ ...current, rooms: previousRooms }));
    setSelectedRoom(previousRooms[0]?.id ?? "");
    setHistoryVersion((version) => version + 1);
  };

  const redoRooms = () => {
    const nextRooms = roomFuture.current.pop();
    if (!nextRooms) return;
    roomHistory.current = [...roomHistory.current, rooms];
    setProject((current) => ({ ...current, rooms: nextRooms }));
    setSelectedRoom(nextRooms[0]?.id ?? "");
    setHistoryVersion((version) => version + 1);
  };

  const resetGeneratedPlan = () => {
    const safeForm = normalizeForm(form);
    const nextRooms = buildPlan(Number(safeForm.bedrooms), Number(safeForm.bathrooms), Number(safeForm.areaSqft));
    updateRooms(nextRooms);
    setSelectedRoom(nextRooms.find((room) => room.id === "living")?.id ?? nextRooms[0]?.id ?? "");
    setZoom(1);
  };

  const acceptPlanFile = (file: File) => {
    const allowedTypes = ["image/png", "image/jpeg", "image/webp", "application/pdf"];
    if (!allowedTypes.includes(file.type)) {
      setUploadAnalysis({ status: "error", message: "Use a PNG, JPG, WEBP or PDF floor plan." });
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setUploadAnalysis({ status: "error", message: "The file must be 20 MB or smaller." });
      return;
    }
    if (uploadedPlan) URL.revokeObjectURL(uploadedPlan.url);
    setUploadedPlan({ name: file.name, size: file.size, type: file.type, url: URL.createObjectURL(file) });
    setUploadAnalysis({ status: "ready" });
    setSection("upload");
  };

  const removeUploadedPlan = () => {
    if (uploadedPlan) URL.revokeObjectURL(uploadedPlan.url);
    setUploadedPlan(null);
    setUploadAnalysis({ status: "idle" });
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const analyzeUploadedPlan = async () => {
    if (!uploadedPlan || !fileInputRef.current?.files?.[0]) {
      setUploadAnalysis({ status: "error", message: "Upload a property plan first." });
      return;
    }
    if (!PLAN_API_URL) {
      setUploadAnalysis({ status: "error", message: "The image-analysis API is not connected yet. Your file stays local and no fake valuation was created." });
      return;
    }

    setUploadAnalysis({ status: "loading" });
    try {
      const body = new FormData();
      body.append("file", fileInputRef.current.files[0]);
      body.append("property", JSON.stringify(form));
      const response = await fetch(`${PLAN_API_URL}/analyze`, { method: "POST", body });
      const result = (await response.json()) as { predicted_price_inr?: number; detail?: string };
      if (!response.ok || typeof result.predicted_price_inr !== "number") {
        throw new Error(result.detail ?? "The image analysis service returned an invalid response.");
      }
      setUploadAnalysis({ status: "success", price: result.predicted_price_inr });
    } catch (error) {
      setUploadAnalysis({ status: "error", message: error instanceof Error ? error.message : "Image analysis failed." });
    }
  };

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(project));
  }, [project]);

  useEffect(() => {
    let cancelled = false;
    const hydrateProject = async () => {
      if (!API_URL) {
        setDatabaseSync("offline");
        setRemoteReady(true);
        return;
      }
      try {
        const response = await fetch(`${API_URL}/project/${PROJECT_ID}`);
        if (response.status === 404) {
          if (!cancelled) {
            setDatabaseSync("syncing");
            setRemoteReady(true);
          }
          return;
        }
        if (!response.ok) throw new Error("Could not load the saved project.");
        const result = (await response.json()) as { project?: ProjectState };
        if (result.project && !cancelled) {
          setProject({
            form: normalizeForm({ ...initialForm, ...result.project.form }),
            rooms: Array.isArray(result.project.rooms) && result.project.rooms.length
              ? result.project.rooms
              : buildPlan(2, 2, 1200),
            prediction: result.project.prediction ?? { status: "idle" },
          });
        }
        if (!cancelled) {
          setDatabaseSync("synced");
          setRemoteReady(true);
        }
      } catch {
        if (!cancelled) {
          setDatabaseSync("offline");
          setRemoteReady(true);
        }
      }
    };
    void hydrateProject();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!remoteReady || !API_URL) return;
    const timer = window.setTimeout(async () => {
      setDatabaseSync("syncing");
      try {
        const response = await fetch(`${API_URL}/project/${PROJECT_ID}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            project,
            upload_metadata: uploadedPlan
              ? { name: uploadedPlan.name, size: uploadedPlan.size, type: uploadedPlan.type }
              : null,
          }),
        });
        if (!response.ok) throw new Error("Project sync failed.");
        setDatabaseSync("synced");
      } catch {
        setDatabaseSync("offline");
      }
    }, 1200);
    return () => window.clearTimeout(timer);
  }, [project, uploadedPlan, remoteReady, syncRetryNonce]);

  useEffect(() => {
    if (databaseSync !== "offline" || !remoteReady) return;
    const timer = window.setTimeout(() => setSyncRetryNonce((value) => value + 1), 5000);
    return () => window.clearTimeout(timer);
  }, [databaseSync, remoteReady]);

  useEffect(() => {
    if (initialPlanCheck.current) {
      initialPlanCheck.current = false;
      return;
    }

    const timer = window.setTimeout(() => {
      const safeForm = normalizeForm(form);
      const nextRooms = buildPlan(Number(safeForm.bedrooms), Number(safeForm.bathrooms), Number(safeForm.areaSqft));
      setProject((current) => ({ ...current, rooms: nextRooms }));
      setSelectedRoom(nextRooms.find((room) => room.id === "living")?.id ?? nextRooms[0]?.id ?? "");
    }, 350);

    return () => window.clearTimeout(timer);
  }, [form.areaSqft, form.bedrooms, form.bathrooms]);

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      if (!API_URL) {
        setApiHealth("offline");
        return;
      }
      try {
        const response = await fetch(`${API_URL}/health`);
        if (!cancelled) setApiHealth(response.ok ? "online" : "offline");
      } catch {
        if (!cancelled) setApiHealth("offline");
      }
    };
    void check();
    return () => {
      cancelled = true;
    };
  }, []);

  const predict = async () => {
    const now = Date.now();
    if (now - lastPredictionRequestAt.current < PREDICTION_COOLDOWN_MS) {
      setProject((current) => ({
        ...current,
        prediction: { status: "error", message: "Please wait a moment before sending another prediction." },
      }));
      return;
    }
    const safeForm = normalizeForm(form);
    const safeAreaSqft = Number(safeForm.areaSqft);
    if (safeAreaSqft < 100 || !safeForm.location.trim()) {
      setProject((current) => ({
        ...current,
        prediction: { status: "error", message: "Enter an area of at least 100 sq ft and a location." },
      }));
      return;
    }
    if (!API_URL) {
      setProject((current) => ({
        ...current,
        prediction: { status: "error", message: "The real model API is not connected yet. No fake price was generated." },
      }));
      return;
    }

    lastPredictionRequestAt.current = now;
    setProject((current) => ({ ...current, prediction: { status: "loading" } }));
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Project-ID": PROJECT_ID },
        body: JSON.stringify({
          area_sqft: safeAreaSqft,
          area_type: safeForm.areaType,
          location: safeForm.location,
          locality: safeForm.locality || undefined,
          society: safeForm.society || undefined,
          bedrooms: optionalNumber(safeForm.bedrooms),
          bathroom: optionalNumber(safeForm.bathrooms),
          balcony: optionalNumber(safeForm.balcony),
          car_parking: optionalNumber(safeForm.carParking),
          floor_num: optionalNumber(safeForm.floorNumber),
          total_floors: optionalNumber(safeForm.totalFloors),
          property_type: safeForm.propertyType,
          furnishing: safeForm.furnishing,
          transaction: safeForm.transaction,
          ownership: safeForm.ownership,
          facing: safeForm.facing || undefined,
          overlooking: safeForm.overlooking || undefined,
        }),
      });
      const result = (await response.json()) as { predicted_price_inr?: number; detail?: string };
      if (!response.ok || typeof result.predicted_price_inr !== "number") {
        throw new Error(result.detail ?? "The prediction service returned an invalid response.");
      }
      setProject((current) => ({
        ...current,
        prediction: { status: "success", price: result.predicted_price_inr! },
      }));
      setApiHealth("online");
    } catch (error) {
      setProject((current) => ({
        ...current,
        prediction: {
          status: "error",
          message: error instanceof Error ? error.message : "Prediction request failed.",
        },
      }));
      setApiHealth("offline");
    }
  };

  const applyTool = (tool: Tool) => {
    setActiveTool(tool);
    if (tool === "select") return;
    if (tool === "room") {
      const nextIndex = rooms.filter((room) => room.id.startsWith("custom-room")).length + 1;
      const newRoom: PlanRoom = {
        id: `custom-room-${Date.now()}`,
        label: `Flex room ${nextIndex}`,
        detail: "10 m²",
        x: 220 + nextIndex * 9,
        y: 260 + nextIndex * 8,
        width: 125,
        height: 92,
        hasDoor: true,
      };
      updateRooms([...rooms, newRoom]);
      setSelectedRoom(newRoom.id);
      setActiveTool("select");
      return;
    }

    const currentRoom = rooms.find((room) => room.id === selectedRoom);
    if (!currentRoom) {
      setActiveTool("select");
      return;
    }
    if (tool === "delete") {
      const nextRooms = rooms.filter((room) => room.id !== selectedRoom);
      updateRooms(nextRooms);
      setSelectedRoom(nextRooms[0]?.id ?? "");
    } else if (tool === "door") {
      updateRooms(rooms.map((room) => (room.id === selectedRoom ? { ...room, hasDoor: !room.hasDoor } : room)));
    } else if (tool === "plant") {
      updateRooms(rooms.map((room) => (room.id === selectedRoom ? { ...room, hasPlant: !room.hasPlant } : room)));
    } else if (tool === "label") {
      const nextLabel = currentRoom.label.startsWith("Custom") ? "Flex space" : `Custom ${currentRoom.label}`;
      updateRooms(rooms.map((room) => (room.id === selectedRoom ? { ...room, label: nextLabel } : room)));
    }
    setActiveTool("select");
  };

  const centerPanel = useMemo(() => {
    if (section === "upload") {
      return (
        <div className="upload-preview-panel">
          <div className="canvas-header">
            <div>
              <h2>AI property valuation</h2>
              <p>Preview the real CAD or floor-plan image before analysis.</p>
            </div>
            {uploadedPlan && <span className="file-type-badge">{uploadedPlan.type === "application/pdf" ? "PDF" : "IMAGE"}</span>}
          </div>
          <div className={`plan-preview ${uploadedPlan ? "has-file" : ""}`}>
            {uploadedPlan ? (
              uploadedPlan.type === "application/pdf" ? (
                <iframe title={uploadedPlan.name} src={uploadedPlan.url} />
              ) : (
                <img src={uploadedPlan.url} alt={`Uploaded property plan ${uploadedPlan.name}`} />
              )
            ) : (
              <div className="empty-preview">
                <FileImage />
                <strong>Your property plan will appear here</strong>
                <span>Upload a clear top-down CAD, blueprint or floor-plan image.</span>
              </div>
            )}
          </div>
          {uploadedPlan && (
            <div className="uploaded-file-bar">
              <FileImage />
              <div><strong>{uploadedPlan.name}</strong><span>{(uploadedPlan.size / 1024 / 1024).toFixed(2)} MB · Ready for analysis</span></div>
              <Button variant="ghost" size="icon" onClick={removeUploadedPlan} aria-label="Remove uploaded plan"><Trash2 /></Button>
            </div>
          )}
        </div>
      );
    }

    if (section === "settings") {
      return (
        <div className="settings-panel">
          <div className="functional-heading">
            <div>
              <h2>Connection settings</h2>
              <p>The workspace saves automatically in this browser.</p>
            </div>
          </div>
          <div className="settings-stack">
            <article>
              <div className="api-state">
                {apiHealth === "online" ? <Wifi /> : <WifiOff />}
                <div>
                  <strong>Prediction API</strong>
                  <span>{API_URL || "Not configured in VITE_PREDICTION_API_URL"}</span>
                </div>
              </div>
              <span className={`connection-badge ${apiHealth}`}>{apiHealth}</span>
            </article>
            <article>
              <div>
                <strong>Automatic browser save</strong>
                <span>Property fields, rooms and the latest prediction persist on this device.</span>
              </div>
              <span className="connection-badge online">active</span>
            </article>
            <article>
              <div className="api-state">
                <Database />
                <div>
                  <strong>Supabase project sync</strong>
                  <span>Project ID: {PROJECT_ID}</span>
                </div>
              </div>
              <span className={`connection-badge ${databaseSync === "synced" ? "online" : ""}`}>{databaseSync}</span>
            </article>
          </div>
        </div>
      );
    }

    return (
      <>
        <div className="canvas-header">
          <div className="canvas-title-block">
            <h2>Visual floor plan</h2>
            <div className="plan-stats" aria-label="Floor plan summary">
              <span><strong>{areaM2}</strong> m²</span>
              <span><strong>{layoutBedrooms}</strong> bedrooms</span>
              <span><strong>{layoutBathrooms}</strong> bathrooms</span>
              <span>Floor <strong>{safeFormView.floorNumber}</strong></span>
            </div>
            <p className="selected-space">{selectedRoomData ? `${selectedRoomData.label} selected · drag to reposition` : `${rooms.length} editable spaces`}</p>
          </div>
          <div className="canvas-actions">
            <div className="canvas-modes">
              <Button className={canvasMode === "2d" ? "dark-button compact" : "soft-button compact"} onClick={() => setCanvasMode("2d")}>2D plan</Button>
              <Button className={canvasMode === "3d" ? "dark-button compact" : "soft-button compact"} onClick={() => setCanvasMode("3d")}>3D view</Button>
              <Button variant="outline" size="icon" className="soft-button icon-button" onClick={() => document.documentElement.requestFullscreen?.()} aria-label="Fullscreen"><Expand /></Button>
            </div>
            <div className="plan-utilities" aria-label="Floor plan controls">
              <Button variant="ghost" size="icon" onClick={undoRooms} disabled={!roomHistory.current.length} title="Undo" aria-label="Undo"><Undo2 /></Button>
              <Button variant="ghost" size="icon" onClick={redoRooms} disabled={!roomFuture.current.length} title="Redo" aria-label="Redo"><Redo2 /></Button>
              <Button variant="ghost" size="icon" className={snapToGrid ? "active" : ""} onClick={() => setSnapToGrid((value) => !value)} title="Snap rooms to grid" aria-label="Toggle snap to grid"><Magnet /></Button>
              <Button variant="ghost" size="icon" className={showGrid ? "active" : ""} onClick={() => setShowGrid((value) => !value)} title="Show grid" aria-label="Toggle grid"><Grid2X2 /></Button>
              <Button variant="ghost" size="icon" className={showDimensions ? "active" : ""} onClick={() => setShowDimensions((value) => !value)} title="Show dimensions" aria-label="Toggle dimensions"><Ruler /></Button>
              <Button variant="ghost" size="icon" onClick={resetGeneratedPlan} title="Reset generated plan" aria-label="Reset generated plan"><RotateCcw /></Button>
            </div>
          </div>
        </div>
        <FloorPlan
          rooms={rooms}
          updateRooms={updateRooms}
          selectedRoom={selectedRoom}
          setSelectedRoom={setSelectedRoom}
          mode={canvasMode}
          zoom={zoom}
          snapToGrid={snapToGrid}
          showGrid={showGrid}
          showDimensions={showDimensions}
        />
        <div className="canvas-toolbar">
          <ToolButton label="Select and move" active={activeTool === "select"} onClick={() => applyTool("select")}><MousePointer2 /></ToolButton>
          <ToolButton label="Toggle door" onClick={() => applyTool("door")}><DoorOpen /></ToolButton>
          <ToolButton label="Add room" onClick={() => applyTool("room")}><Layers3 /></ToolButton>
          <ToolButton label="Rename room" onClick={() => applyTool("label")}><Type /></ToolButton>
          <ToolButton label="Toggle plant" onClick={() => applyTool("plant")}><TreePine /></ToolButton>
          <Separator orientation="vertical" className="toolbar-divider" />
          <ToolButton label="Delete room" destructive onClick={() => applyTool("delete")}><Trash2 /></ToolButton>
        </div>
        <div className="zoom-controls">
          <Button variant="ghost" size="icon-xs" onClick={() => setZoom((value) => Math.max(0.65, value - 0.1))}><ZoomOut /></Button>
          <span>{Math.round(zoom * 100)}%</span>
          <Button variant="ghost" size="icon-xs" onClick={() => setZoom((value) => Math.min(1.35, value + 0.1))}><ZoomIn /></Button>
        </div>
      </>
    );
  }, [section, uploadedPlan, apiHealth, databaseSync, areaM2, safeFormView.floorNumber, rooms, canvasMode, selectedRoom, selectedRoomData, layoutBedrooms, layoutBathrooms, zoom, activeTool, snapToGrid, showGrid, showDimensions, historyVersion]);

  return (
    <main className="app-shell">
      <Card className="topbar card">
        <div className="brand">
          <img src="/favicon.svg" alt="" />
          <span>SpaceMap</span>
        </div>
      </Card>

      <div className="workspace">
        <nav className="rail card" aria-label="Primary navigation">
          <Tooltip>
            <TooltipTrigger render={<Button variant="ghost" size="icon" className={`rail-item ${section === "plan" ? "active" : ""}`} onClick={() => setSection("plan")} aria-label="Floor plan" />}>
              <Layers3 />
            </TooltipTrigger>
            <TooltipContent side="right">Floor plan</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger render={<Button variant="ghost" size="icon" className={`rail-item ${section === "upload" ? "active" : ""}`} onClick={() => setSection("upload")} aria-label="Upload property plan" />}>
              <Upload />
            </TooltipTrigger>
            <TooltipContent side="right">Upload property plan</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger render={<Button variant="ghost" size="icon" className={`rail-item push-bottom ${section === "settings" ? "active" : ""}`} onClick={() => setSection("settings")} aria-label="Settings" />}>
              <Settings />
            </TooltipTrigger>
            <TooltipContent side="right">Settings</TooltipContent>
          </Tooltip>
        </nav>

        <Card className={`details-panel card ${section === "upload" ? "upload-sidebar" : ""}`}>
          {section === "upload" ? (
            <>
              <div className="panel-heading">
                <div><span className="eyebrow">Image valuation</span><h1>Upload property plan</h1></div>
              </div>
              <input ref={fileInputRef} className="file-input" type="file" accept="image/png,image/jpeg,image/webp,application/pdf" onChange={(event) => { const file = event.target.files?.[0]; if (file) acceptPlanFile(file); }} />
              <button
                type="button"
                className={`upload-dropzone ${draggingPlan ? "dragging" : ""}`}
                onClick={() => fileInputRef.current?.click()}
                onDragEnter={(event) => { event.preventDefault(); setDraggingPlan(true); }}
                onDragOver={(event) => event.preventDefault()}
                onDragLeave={() => setDraggingPlan(false)}
                onDrop={(event) => { event.preventDefault(); setDraggingPlan(false); const file = event.dataTransfer.files?.[0]; if (file) acceptPlanFile(file); }}
              >
                <span className="upload-icon"><Upload /></span>
                <strong>{uploadedPlan ? "Replace this plan" : "Drop your CAD here"}</strong>
                <span>or click to browse files</span>
                <em>PNG, JPG, WEBP or PDF · max 20 MB</em>
              </button>
              <div className="upload-tips">
                <strong>For the clearest analysis</strong>
                <span>✓ Use a straight top-down floor plan</span>
                <span>✓ Keep dimensions and room labels visible</span>
                <span>✓ Avoid blurry photos, glare and shadows</span>
              </div>
              <p className="privacy-note">Uploads remain in your browser until an image-analysis API is configured.</p>
            </>
          ) : (
            <>
              <div className="panel-heading">
                <div><span className="eyebrow">Prediction inputs</span><h1>Property details</h1></div>
                <span className="auto-badge"><Sparkles /> Auto plan</span>
              </div>
              <div className="details-form">
            <Field label="Area (sq ft)"><SafeNumberInput field="areaSqft" value={form.areaSqft} onChange={(value) => updateForm("areaSqft", value)} onBlur={normalizeCurrentForm} /></Field>
            <Field label="Area type"><Select value={form.areaType} onValueChange={(value) => updateForm("areaType", value as PropertyForm["areaType"])}><SelectTrigger className="w-full"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="super">Super area</SelectItem><SelectItem value="carpet">Carpet area</SelectItem></SelectContent></Select></Field>
            <Field label="Location"><span className="input-with-icon"><Input value={form.location} onChange={(event) => updateForm("location", event.target.value)} onBlur={normalizeCurrentForm} /><button type="button" onClick={() => setMapOpen(true)} aria-label="Open location on map"><MapPin /></button></span></Field>
            <Field label="Locality"><Input value={form.locality} onChange={(event) => updateForm("locality", event.target.value)} /></Field>
            <Field label="Society" wide><Input value={form.society} onChange={(event) => updateForm("society", event.target.value)} /></Field>
            <Field label="Bedrooms"><SafeNumberInput field="bedrooms" value={form.bedrooms} onChange={(value) => updateForm("bedrooms", value)} onBlur={normalizeCurrentForm} /></Field>
            <Field label="Bathrooms"><SafeNumberInput field="bathrooms" value={form.bathrooms} onChange={(value) => updateForm("bathrooms", value)} onBlur={normalizeCurrentForm} /></Field>
            <Field label="Balconies"><SafeNumberInput field="balcony" value={form.balcony} onChange={(value) => updateForm("balcony", value)} onBlur={normalizeCurrentForm} /></Field>
            <Field label="Parking spaces"><SafeNumberInput field="carParking" value={form.carParking} onChange={(value) => updateForm("carParking", value)} onBlur={normalizeCurrentForm} /></Field>
            <Field label="Current floor"><SafeNumberInput field="floorNumber" value={form.floorNumber} onChange={(value) => updateForm("floorNumber", value)} onBlur={normalizeCurrentForm} /></Field>
            <Field label="Total floors"><SafeNumberInput field="totalFloors" value={form.totalFloors} onChange={(value) => updateForm("totalFloors", value)} onBlur={normalizeCurrentForm} /></Field>
            <Field label="Property type"><Select value={form.propertyType} onValueChange={(value) => updateForm("propertyType", value as PropertyForm["propertyType"])}><SelectTrigger className="w-full"><SelectValue /></SelectTrigger><SelectContent>{["flat", "villa", "house", "builder_floor", "penthouse", "studio", "plot", "unknown"].map((value) => <SelectItem key={value} value={value}>{readable(value)}</SelectItem>)}</SelectContent></Select></Field>
            <Field label="Furnishing"><Select value={form.furnishing} onValueChange={(value) => updateForm("furnishing", value as PropertyForm["furnishing"])}><SelectTrigger className="w-full"><SelectValue /></SelectTrigger><SelectContent>{["unfurnished", "semi_furnished", "furnished", "unknown"].map((value) => <SelectItem key={value} value={value}>{readable(value)}</SelectItem>)}</SelectContent></Select></Field>
            <Field label="Transaction"><Select value={form.transaction} onValueChange={(value) => updateForm("transaction", value as PropertyForm["transaction"])}><SelectTrigger className="w-full"><SelectValue /></SelectTrigger><SelectContent>{["resale", "new_property", "other", "unknown"].map((value) => <SelectItem key={value} value={value}>{readable(value)}</SelectItem>)}</SelectContent></Select></Field>
            <Field label="Ownership"><Select value={form.ownership} onValueChange={(value) => updateForm("ownership", value as PropertyForm["ownership"])}><SelectTrigger className="w-full"><SelectValue /></SelectTrigger><SelectContent>{["freehold", "cooperative_society", "leasehold", "unknown"].map((value) => <SelectItem key={value} value={value}>{readable(value)}</SelectItem>)}</SelectContent></Select></Field>
            <Field label="Facing"><Input value={form.facing} onChange={(event) => updateForm("facing", event.target.value)} /></Field>
            <Field label="Overlooking"><Input value={form.overlooking} onChange={(event) => updateForm("overlooking", event.target.value)} /></Field>
              </div>
            </>
          )}
        </Card>

        <Card className="canvas-panel card">{centerPanel}</Card>

        <aside className="summary-column">
          {section === "upload" ? (
            <>
              <Card className="summary-card card upload-summary">
                <span className="eyebrow">Upload summary</span>
                <h2>Plan details</h2>
                <dl>
                  <div><dt>File</dt><dd>{uploadedPlan?.name ?? "No file"}</dd></div>
                  <div><dt>Format</dt><dd>{uploadedPlan ? uploadedPlan.type.split("/").pop()?.toUpperCase() : "—"}</dd></div>
                  <div><dt>Size</dt><dd>{uploadedPlan ? `${(uploadedPlan.size / 1024 / 1024).toFixed(2)} MB` : "—"}</dd></div>
                  <div><dt>Location</dt><dd>{safeFormView.location}</dd></div>
                </dl>
              </Card>
              <Card className={`prediction-card upload-analysis-card card ${uploadAnalysis.status}`}>
                <div className="prediction-heading">
                  <div><span className="eyebrow">Vision valuation</span><h2>Plan analysis</h2></div>
                  <FileImage />
                </div>
                <div className="prediction-value">
                  {uploadAnalysis.status === "success" ? (
                    <><strong>{formatCurrency(uploadAnalysis.price)}</strong><span>Returned by the configured image-analysis API.</span></>
                  ) : uploadAnalysis.status === "loading" ? (
                    <><strong>Analyzing…</strong><span>Reading the uploaded plan and property context.</span></>
                  ) : (
                    <strong className="prediction-placeholder" aria-label="No image valuation yet">—</strong>
                  )}
                </div>
                {uploadAnalysis.status === "error" && <p className="prediction-error">{uploadAnalysis.message}</p>}
                <div className="analysis-checks"><span>Room geometry</span><span>Visible dimensions</span><span>Property context</span></div>
                <Button className="dark-button predict-button" onClick={analyzeUploadedPlan} disabled={!uploadedPlan || uploadAnalysis.status === "loading"}>
                  <Sparkles /> {uploadAnalysis.status === "loading" ? "Analyzing plan…" : "Analyze uploaded plan"}
                </Button>
                <p className="honesty-note">No sample valuation is shown. Results require a real image-analysis endpoint.</p>
              </Card>
            </>
          ) : (
            <>
          <Card className="summary-card card">
            <span className="eyebrow">Live summary</span>
            <h2>Property snapshot</h2>
            <dl>
              <div><dt>Area</dt><dd>{areaSqft ? `${areaSqft.toLocaleString("en-IN")} sq ft` : "—"}</dd></div>
              <div><dt>Configuration</dt><dd>{safeFormView.bedrooms} bed · {safeFormView.bathrooms} bath</dd></div>
              <div><dt>Property</dt><dd>{readable(form.propertyType)}</dd></div>
              <div><dt>Location</dt><dd>{safeFormView.location}</dd></div>
              <div><dt>Floor</dt><dd>{safeFormView.floorNumber} / {safeFormView.totalFloors}</dd></div>
            </dl>
          </Card>

          <Card className={`prediction-card card ${prediction.status}`}>
            <div className="prediction-heading">
              <div>
                <span className="eyebrow">AI price prediction</span>
                <h2>Estimated value</h2>
              </div>
              <Box />
            </div>
            <div className="prediction-value">
              {prediction.status === "success" ? (
                <>
                  <strong>{formatCurrency(prediction.price)}</strong>
                  {pricePerSqft && <span>{formatCurrency(pricePerSqft)} per sq ft</span>}
                </>
              ) : prediction.status === "loading" ? (
                <><strong>Calculating…</strong><span>Running the real 90.64% R² model</span></>
              ) : (
                <strong className="prediction-placeholder" aria-label="No prediction yet">—</strong>
              )}
            </div>
            {prediction.status === "error" && <p className="prediction-error">{prediction.message}</p>}
            <div className="model-score"><span>Validated model score</span><strong>90.64% R²</strong></div>
            <Button className="dark-button predict-button" onClick={predict} disabled={prediction.status === "loading"}>
              <Sparkles /> {prediction.status === "loading" ? "Predicting price…" : "Predict price"}
            </Button>
          </Card>
            </>
          )}
        </aside>
      </div>

      {mapOpen && (
        <div className="map-overlay" role="presentation" onMouseDown={() => setMapOpen(false)}>
          <section className="map-dialog card" role="dialog" aria-modal="true" aria-labelledby="map-title" onMouseDown={(event) => event.stopPropagation()}>
            <header>
              <div>
                <span className="eyebrow">Real map preview</span>
                <h2 id="map-title">{mapQuery}</h2>
              </div>
              <Button variant="ghost" size="icon" onClick={() => setMapOpen(false)} aria-label="Close map"><X /></Button>
            </header>
            <iframe title={`Map of ${mapQuery}`} src={mapEmbedUrl} loading="lazy" referrerPolicy="no-referrer-when-downgrade" />
            <footer>
              <p><MapPinned /> Map follows the location and locality fields.</p>
              <Button variant="outline" onClick={() => window.open(mapPageUrl, "_blank", "noopener,noreferrer")}><ExternalLink /> Open in Google Maps</Button>
            </footer>
          </section>
        </div>
      )}
    </main>
  );
}

export default App;
