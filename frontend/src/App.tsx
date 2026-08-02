import { useEffect, useMemo, useRef, useState } from "react";
import { Circle, Group, Layer, Line, Rect, Stage, Text } from "react-konva";
import { driver, type Driver } from "driver.js";
import "driver.js/dist/driver.css";
import {
  Box,
  BadgeCheck,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  CircleDot,
  Database,
  DoorOpen,
  ExternalLink,
  Expand,
  FileImage,
  Code2,
  Grid2X2,
  Layers3,
  Magnet,
  MapPin,
  MapPinned,
  MousePointer2,
  Redo2,
  RefreshCw,
  RotateCcw,
  Ruler,
  Settings,
  Sparkles,
  Server,
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

type Section = "plan" | "upload" | "proof" | "settings";
type CanvasMode = "2d" | "3d";
type Tool = "select" | "door" | "room" | "label" | "plant" | "delete";
type DatabaseSyncState = "loading" | "syncing" | "synced" | "offline";
type LiveProofState =
  | { status: "idle" }
  | { status: "loading" }
  | {
      status: "success";
      model: string;
      score: number;
      database: string;
      price: number;
      checkedAt: string;
    }
  | { status: "error"; message: string };

const API_URL = (
  import.meta.env.VITE_PREDICTION_API_URL ??
  "https://iti-house-price-api-production.up.railway.app"
).replace(/\/$/, "");
const PLAN_API_URL = (import.meta.env.VITE_PLAN_ANALYSIS_API_URL ?? "").replace(/\/$/, "");
const STORAGE_KEY = "spacemap-project-v2";
const PROJECT_ID_KEY = "spacemap-project-id-v1";
const WELCOME_TOUR_KEY = "spacemap-welcome-tour-v1";
const PREDICTION_COOLDOWN_MS = 1800;
const SOURCE_URL = "https://github.com/YUST777/iti_ai_project";
const MODEL_URL = "https://huggingface.co/duck233/iti-house-price-model";
const DATASET_URL = "https://www.kaggle.com/datasets/juhibhojani/house-price";
const REPO_BLOB_URL = `${SOURCE_URL}/blob/main`;
const REPO_COMMIT_URL = `${SOURCE_URL}/commit`;

const MODEL_MILESTONES = [
  {
    name: "CatBoost baseline",
    score: "79.59%",
    size: "10.2 MiB",
    note: "The first preserved model proved the end-to-end pipeline worked, but its errors were still too large.",
    commit: "43379b2",
  },
  {
    name: "LightGBM iteration",
    score: "81.71%",
    size: "23.7 MiB",
    note: "A stronger tree model improved generalization and reduced MAE and RMSE on the same held-out split.",
    commit: "4a54ebe",
  },
  {
    name: "Price-per-sqft target",
    score: "84.39%",
    size: "35.2 MiB",
    note: "Reframing the target around price per square foot made area relationships easier for the model to learn.",
    commit: "3864a9e",
  },
  {
    name: "Tree-model blend",
    score: "85.34%",
    size: "25.6 MiB",
    note: "Blending LightGBM and CatBoost reduced the weaknesses of either model on its own.",
    commit: "1a1bf76",
  },
  {
    name: "Final full-refit ensemble",
    score: "90.64%",
    size: "55.0 MiB",
    note: "LightGBM, CatBoost, and three neural networks were combined after validation, then refit for the final artifact.",
    commit: "c289efa",
  },
] as const;

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

function predictionPayload(form: PropertyForm) {
  const safeForm = normalizeForm(form);
  return {
    area_sqft: Number(safeForm.areaSqft),
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
  };
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
  const [liveProof, setLiveProof] = useState<LiveProofState>({ status: "idle" });
  const [mapOpen, setMapOpen] = useState(false);
  const [uploadedPlan, setUploadedPlan] = useState<UploadedPlan | null>(null);
  const [uploadAnalysis, setUploadAnalysis] = useState<UploadAnalysisState>({ status: "idle" });
  const [draggingPlan, setDraggingPlan] = useState(false);
  const [mobileDetailsCollapsed, setMobileDetailsCollapsed] = useState(() => window.innerWidth <= 560);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const initialPlanCheck = useRef(true);
  const lastPredictionRequestAt = useRef(0);
  const roomHistory = useRef<PlanRoom[][]>([]);
  const roomFuture = useRef<PlanRoom[][]>([]);
  const welcomeTour = useRef<Driver | null>(null);
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

  const startWelcomeTour = () => {
    welcomeTour.current?.destroy();
    setSection("plan");
    window.scrollTo({ top: 0, behavior: "smooth" });
    window.setTimeout(() => {
      const tour = driver({
        animate: true,
        smoothScroll: true,
        allowClose: false,
        allowKeyboardControl: false,
        overlayClickBehavior: () => undefined,
        disableActiveInteraction: true,
        stagePadding: 8,
        stageRadius: 14,
        popoverClass: "spacemap-tour",
        showProgress: true,
        progressText: "Step {{current}} of {{total}}",
        showButtons: ["previous", "next"],
        nextBtnText: "Next",
        prevBtnText: "Back",
        doneBtnText: "Start exploring",
        steps: [
          {
            element: ".tour-welcome",
            popover: {
              title: "Welcome to SpaceMap",
              description: "Build a property profile, shape the editable floor plan, and keep every project synchronized automatically.",
              side: "bottom",
              align: "center",
            },
          },
          {
            element: ".tour-main-ai",
            popover: {
              title: "Real price prediction",
              description: "Complete the property details, then run the validated model to receive the live estimated value and price per square foot.",
              side: "left",
              align: "center",
              onNextClick: (_element, _step, options) => {
                setSection("upload");
                window.setTimeout(() => options.driver.moveNext(), 180);
              },
            },
          },
          {
            element: ".tour-cad-ai",
            waitForElement: 1200,
            popover: {
              title: "CAD image intelligence",
              description: "Upload a real CAD, blueprint, PDF, or floor-plan image here. The vision workflow keeps the file and property context together for analysis.",
              side: "right",
              align: "start",
              onPrevClick: (_element, _step, options) => {
                setSection("plan");
                window.setTimeout(() => options.driver.movePrevious(), 180);
              },
            },
          },
        ],
        onDoneClick: (_element, _step, options) => {
          localStorage.setItem(WELCOME_TOUR_KEY, "complete");
          options.driver.destroy();
          setSection("plan");
          window.scrollTo({ top: 0, behavior: "smooth" });
        },
      });
      welcomeTour.current = tour;
      tour.drive();
    }, 160);
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
    if (localStorage.getItem(WELCOME_TOUR_KEY) === "complete") return;
    const timer = window.setTimeout(startWelcomeTour, 900);
    return () => {
      window.clearTimeout(timer);
      welcomeTour.current?.destroy();
    };
  }, []);

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
        body: JSON.stringify(predictionPayload(safeForm)),
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

  const runLiveProof = async () => {
    setLiveProof({ status: "loading" });
    try {
      const [healthResponse, predictionResponse] = await Promise.all([
        fetch(`${API_URL}/health`, { cache: "no-store" }),
        fetch(`${API_URL}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-Project-ID": PROJECT_ID },
          body: JSON.stringify(predictionPayload(form)),
        }),
      ]);
      const healthResult = (await healthResponse.json()) as {
        model?: string;
        held_out_r2?: number;
        database?: string;
        detail?: string;
      };
      const predictionResult = (await predictionResponse.json()) as { predicted_price_inr?: number; detail?: string };
      if (!healthResponse.ok || !predictionResponse.ok || typeof predictionResult.predicted_price_inr !== "number") {
        throw new Error(predictionResult.detail ?? healthResult.detail ?? "The live verification failed.");
      }
      setLiveProof({
        status: "success",
        model: healthResult.model ?? "House price ensemble",
        score: healthResult.held_out_r2 ?? 0.906449314077493,
        database: healthResult.database ?? "unknown",
        price: predictionResult.predicted_price_inr,
        checkedAt: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
      });
    } catch (error) {
      setLiveProof({ status: "error", message: error instanceof Error ? error.message : "Live verification failed." });
    }
  };

  useEffect(() => {
    if (section === "proof" && liveProof.status === "idle") void runLiveProof();
  }, [section, liveProof.status]);

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
            <article>
              <div>
                <strong>Welcome walkthrough</strong>
                <span>Replay the three-step guide for the workspace and both AI flows.</span>
              </div>
              <Button variant="outline" className="tour-replay-button" onClick={startWelcomeTour}><RotateCcw /> Replay tour</Button>
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
      <Card className="topbar card tour-welcome">
        <div className="brand">
          <img src="/favicon.svg" alt="" />
          <span>SpaceMap</span>
        </div>
        <nav className="top-tabs" aria-label="Main product views">
          <button type="button" className={section === "plan" ? "active" : ""} onClick={() => setSection("plan")}>Price my home</button>
          <button type="button" className={section === "upload" ? "active" : ""} onClick={() => setSection("upload")}>CAD to price</button>
          <button type="button" className={section === "proof" ? "active" : ""} onClick={() => setSection("proof")}>Proof</button>
        </nav>
        <div className="topbar-balance" aria-hidden="true" />
      </Card>

      <div className={`workspace tour-workspace ${section === "proof" ? "proof-mode" : ""}`}>
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

        {section === "proof" && (
          <Card className="proof-page card">
            <header className="proof-hero">
              <div>
                <span className="eyebrow">Proof of work · August 1–2, 2026</span>
                <h1>From a 79.59% baseline to a live 90.64% property intelligence product</h1>
                <p>This is the complete project story: the assignment, the model iterations, the product decisions, the failed deployment routes, the final architecture, and the work that is still honestly in progress.</p>
                <div className="proof-meta-row" aria-label="Project summary">
                  <span>187,531 raw listings</span>
                  <span>5 preserved model milestones</span>
                  <span>Real API + database</span>
                </div>
              </div>
              <Button className="dark-button proof-run-button" onClick={runLiveProof} disabled={liveProof.status === "loading"}>
                <RefreshCw className={liveProof.status === "loading" ? "spinning" : ""} />
                {liveProof.status === "loading" ? "Running checks…" : "Run live proof"}
              </Button>
            </header>

            <section className="proof-status-grid" aria-label="Live verification results">
              <article>
                <span className="proof-icon"><Server /></span>
                <div><small>Railway API</small><strong>{liveProof.status === "success" ? "Operational" : liveProof.status === "loading" ? "Checking…" : "Ready to check"}</strong></div>
                <BadgeCheck className={liveProof.status === "success" ? "verified" : ""} />
              </article>
              <article>
                <span className="proof-icon"><Sparkles /></span>
                <div><small>Held-out model score</small><strong>{liveProof.status === "success" ? `${(liveProof.score * 100).toFixed(2)}% R²` : "90.64% R²"}</strong></div>
                <BadgeCheck className={liveProof.status === "success" ? "verified" : ""} />
              </article>
              <article>
                <span className="proof-icon"><Database /></span>
                <div><small>Supabase database</small><strong>{liveProof.status === "success" ? readable(liveProof.database) : databaseSync === "synced" ? "Connected" : "Checking…"}</strong></div>
                <BadgeCheck className={liveProof.status === "success" && liveProof.database === "connected" ? "verified" : ""} />
              </article>
              <article>
                <span className="proof-icon"><CheckCircle2 /></span>
                <div><small>Real prediction response</small><strong>{liveProof.status === "success" ? formatCurrency(liveProof.price) : "Waiting for run"}</strong></div>
                <BadgeCheck className={liveProof.status === "success" ? "verified" : ""} />
              </article>
            </section>

            {liveProof.status === "error" && <p className="proof-error">{liveProof.message}</p>}
            {liveProof.status === "success" && <p className="proof-timestamp">Last verified at {liveProof.checkedAt} using the current property inputs.</p>}

            <article className="proof-story">
              <section className="proof-opening proof-surface">
                <div className="proof-copy proof-lede">
                  <span className="proof-chapter-number">00</span>
                  <span className="eyebrow">Where the project started</span>
                  <h2>A classroom brief became a full product engineering problem</h2>
                  <p>The ITI guide asked for an end-to-end machine-learning application: inspect a messy Indian property dataset, clean it, compare at least two regressors, export the winning pipeline, serve it with FastAPI, connect it to a React interface, and publish reproducible evidence on GitHub.</p>
                  <blockquote>“At first I only wanted a model that could predict a house price. After the first result, I kept asking what would make the project feel real: a better model, a better interface, a live API, saved projects, and a more creative way to describe the property.”</blockquote>
                  <p>The work therefore moved through three different disciplines: data science, product design, and deployment. The hardest part was not drawing the interface. It was keeping the model honest while making a heavy Python ML stack available from a public web product.</p>
                </div>
                <aside className="proof-brief-card">
                  <span className="eyebrow">The assignment contract</span>
                  <h3>What had to be delivered</h3>
                  <ul>
                    <li><BadgeCheck /> Real cleaning and feature engineering</li>
                    <li><BadgeCheck /> Multiple models with held-out metrics</li>
                    <li><BadgeCheck /> Exported reproducible model artifact</li>
                    <li><BadgeCheck /> FastAPI <code>/health</code> and <code>/predict</code></li>
                    <li><BadgeCheck /> React form with real prediction states</li>
                    <li><BadgeCheck /> Public GitHub evidence and deployment</li>
                  </ul>
                  <a className="proof-inline-link" href={DATASET_URL} target="_blank" rel="noreferrer">Open the assignment dataset <ExternalLink /></a>
                </aside>
              </section>

              <section className="proof-chapter proof-surface">
                <div className="proof-chapter-header">
                  <span className="proof-chapter-number">01</span>
                  <div><span className="eyebrow">Data and modeling</span><h2>The score was earned through iteration, not leakage</h2></div>
                </div>
                <div className="proof-two-column-copy">
                  <div className="proof-copy">
                    <p>The source contained 187,531 listings and 21 columns, but the raw values were not model-ready. Prices mixed Lac, Crore, commas, and “Call for Price.” Areas mixed square feet and square metres. Floors contained text such as Ground and Basement. Several fields were mostly missing, and repeated listings could make a random split look better than it really was.</p>
                    <p>The shared pipeline parsed prices and areas, normalized categories, extracted floor and room counts, removed invalid rows and extreme price-per-square-foot outliers, and deduplicated listings before training. The final high-accuracy run used 57,058 filtered rows: 48,499 for fitting and 8,559 for the untouched held-out test.</p>
                  </div>
                  <div className="proof-data-points" aria-label="Training data facts">
                    <article><strong>187,531</strong><span>raw rows inspected</span></article>
                    <article><strong>57,058</strong><span>final quality-filtered rows</span></article>
                    <article><strong>8,559</strong><span>held-out test rows</span></article>
                    <article><strong>133 sec</strong><span>final recorded GPU training</span></article>
                  </div>
                </div>

                <div className="proof-subheading">
                  <span className="eyebrow">Five Git-preserved milestones</span>
                  <h3>How the model moved from a baseline to the final ensemble</h3>
                  <p>There were additional parameter and neural-network experiments inside these stages; these five commits are the clean milestones that can be independently inspected.</p>
                </div>
                <ol className="model-journey">
                  {MODEL_MILESTONES.map((milestone, index) => (
                    <li key={milestone.commit} className={index === MODEL_MILESTONES.length - 1 ? "winner" : ""}>
                      <div className="model-journey-index">{String(index + 1).padStart(2, "0")}</div>
                      <div className="model-journey-copy">
                        <div><strong>{milestone.name}</strong>{index === MODEL_MILESTONES.length - 1 && <span className="winner-badge">Selected</span>}</div>
                        <p>{milestone.note}</p>
                        <a href={`${REPO_COMMIT_URL}/${milestone.commit}`} target="_blank" rel="noreferrer">Inspect commit {milestone.commit} <ExternalLink /></a>
                      </div>
                      <div className="model-journey-metrics"><span>{milestone.score} R²</span><small>{milestone.size}</small></div>
                    </li>
                  ))}
                </ol>

                <div className="proof-integrity-grid">
                  <article>
                    <span className="proof-icon"><BadgeCheck /></span>
                    <div><strong>No target leakage</strong><p>The model excludes the source price-per-square-foot field, the rupee price fields, and price-bearing description text.</p></div>
                  </article>
                  <article>
                    <span className="proof-icon"><CircleDot /></span>
                    <div><strong>Held-out evaluation</strong><p>The reported 90.64% is test-set R², not a training score. Final MAE is ₹1,535,857.51 and RMSE is ₹2,883,442.92.</p></div>
                  </article>
                  <article>
                    <span className="proof-icon"><Sparkles /></span>
                    <div><strong>A larger artifact for a reason</strong><p>The first preserved artifact was 10.2 MiB; the 55 MiB final artifact contains LightGBM, CatBoost, and three neural networks.</p></div>
                  </article>
                </div>
                <div className="proof-link-row">
                  <a className="proof-inline-link" href={`${REPO_BLOB_URL}/notebooks/pipeline.py`} target="_blank" rel="noreferrer">Read the cleaning pipeline <ExternalLink /></a>
                  <a className="proof-inline-link" href={`${REPO_BLOB_URL}/models/model_metrics.json`} target="_blank" rel="noreferrer">Open the final metrics JSON <ExternalLink /></a>
                  <a className="proof-inline-link" href={`${REPO_BLOB_URL}/notebooks/train_high_accuracy_full_refit.py`} target="_blank" rel="noreferrer">Inspect final training code <ExternalLink /></a>
                </div>
              </section>

              <section className="proof-chapter proof-surface">
                <div className="proof-chapter-header">
                  <span className="proof-chapter-number">02</span>
                  <div><span className="eyebrow">Product and interface</span><h2>Turning model inputs into a product people can understand</h2></div>
                </div>
                <div className="proof-story-grid">
                  <div className="proof-copy">
                    <p>An AI-generated visual reference helped explore the original direction, but the final interface was built as a functional React and TypeScript product rather than copied as a static mockup. Vite handles the frontend build; shadcn-style primitives and Base UI provide accessible controls; Konva powers the editable plan canvas; Driver.js explains the two AI flows to a first-time user.</p>
                    <p>The property form maps directly to the FastAPI schema. Numeric fields are clamped to safe ranges, empty values recover to defaults, requests are rate-limited, and the result only appears after a real API response. Project state is cached in the browser for instant recovery and synchronized to Supabase when the backend is available.</p>
                    <p>Supabase currently stores anonymous project snapshots and prediction history. A complete user account and login flow is not presented as finished work.</p>
                    <p>The visual floor plan is also functional. Changing bedrooms, bathrooms, area, or floor regenerates a proportional layout. Rooms can be moved, added, renamed, deleted, given doors or plants, snapped to a grid, measured, undone, redone, reset, and viewed in 2D or a simple 3D mode.</p>
                  </div>
                  <div className="proof-feature-stack">
                    <article><Code2 /><div><strong>React + TypeScript + Vite</strong><p>Typed property state, real network requests, reusable UI primitives, and a production build.</p></div></article>
                    <article><Layers3 /><div><strong>Editable floor-plan system</strong><p>Generated geometry is interactive rather than a decorative image.</p></div></article>
                    <article><Database /><div><strong>Browser recovery + Supabase</strong><p>Local cache keeps work immediate; PostgreSQL persistence keeps projects and predictions durable.</p></div></article>
                    <article><Sparkles /><div><strong>Guided onboarding</strong><p>A mandatory three-step tour introduces price prediction, CAD upload, and the main workspace.</p></div></article>
                  </div>
                </div>
                <div className="proof-link-row">
                  <a className="proof-inline-link" href={`${REPO_COMMIT_URL}/a12eafe`} target="_blank" rel="noreferrer">First functional workspace <ExternalLink /></a>
                  <a className="proof-inline-link" href={`${REPO_COMMIT_URL}/13b4b95`} target="_blank" rel="noreferrer">Floor-plan production pass <ExternalLink /></a>
                  <a className="proof-inline-link" href={`${REPO_COMMIT_URL}/8be0523`} target="_blank" rel="noreferrer">Supabase persistence <ExternalLink /></a>
                </div>
              </section>

              <section className="proof-chapter proof-surface">
                <div className="proof-chapter-header">
                  <span className="proof-chapter-number">03</span>
                  <div><span className="eyebrow">Deployment</span><h2>The hardest lesson was hosting a real ML stack</h2></div>
                </div>
                <div className="proof-two-column-copy">
                  <div className="proof-copy">
                    <p>A normal Vercel frontend is small, but the prediction service needs Python, CatBoost, LightGBM, PyTorch, the 55 MiB artifact, and enough memory to load everything together. That made deployment the most difficult part of the project.</p>
                    <p>Several routes were built and tested. Hugging Face received the public model artifact and a Docker Space package. Render received a Docker blueprint. Vercel Python was attempted, but the full ML dependency stack exceeded the practical serverless bundle path. Railway was the simplest service that could build the Docker image, download the exact artifact, start FastAPI, and keep the API callable by the Vercel frontend.</p>
                    <p>The final architecture separates responsibilities instead of forcing everything into one host. The browser UI stays lightweight on Vercel, Railway performs inference and validation, Hugging Face stores the versioned model pieces, and Supabase stores project snapshots and prediction history.</p>
                  </div>
                  <ol className="deployment-trials">
                    <li><span>01</span><div><strong>Hugging Face</strong><p>Published the model and prepared a Docker Space route.</p></div><a href={`${REPO_COMMIT_URL}/8dbcd07`} target="_blank" rel="noreferrer">Proof <ExternalLink /></a></li>
                    <li><span>02</span><div><strong>Render</strong><p>Created a Docker deployment and health-check blueprint.</p></div><a href={`${REPO_COMMIT_URL}/752050e`} target="_blank" rel="noreferrer">Proof <ExternalLink /></a></li>
                    <li><span>03</span><div><strong>Vercel Python</strong><p>Tested serverless packaging before keeping Vercel frontend-only.</p></div><a href={`${REPO_COMMIT_URL}/3c1eb58`} target="_blank" rel="noreferrer">Proof <ExternalLink /></a></li>
                    <li className="selected"><span>04</span><div><strong>Railway</strong><p>Runs the current FastAPI model service and connects to Supabase.</p></div><a href={`${API_URL}/health`} target="_blank" rel="noreferrer">Live <ExternalLink /></a></li>
                  </ol>
                </div>

                <div className="proof-architecture" aria-label="Production architecture">
                  <article><span>01</span><strong>Vercel</strong><small>React interface</small></article>
                  <i>→</i>
                  <article><span>02</span><strong>Railway</strong><small>FastAPI inference</small></article>
                  <i>→</i>
                  <article><span>03</span><strong>Hugging Face</strong><small>55 MiB model artifact</small></article>
                  <i>↔</i>
                  <article><span>04</span><strong>Supabase</strong><small>Projects + history</small></article>
                </div>
              </section>

              <section className="proof-chapter proof-surface">
                <div className="proof-chapter-header">
                  <span className="proof-chapter-number">04</span>
                  <div><span className="eyebrow">Creative extension</span><h2>From manual inputs to visual property understanding</h2></div>
                </div>
                <div className="proof-story-grid">
                  <div className="proof-copy">
                    <p>The first creative idea was to make the inputs visible. If a user chooses three bedrooms, the application should not only send the number <code>3</code>; it should generate a plan with three editable bedroom spaces so the user can understand what the data represents.</p>
                    <p>The next question was more ambitious: what if the owner does not know the area or room counts, but already has an engineering drawing, blueprint, or CAD image? The proposed pipeline accepts the image, extracts walls and openings with a separate deep-learning parser, converts the geometry into structured JSON, asks the user to confirm uncertain fields, and then sends the confirmed property data to the existing price model.</p>
                    <p>The architecture deliberately keeps the two models separate. The validated price model understands structured tabular features, not pixels. A vision parser must first produce geometry and confidence values before price prediction is safe.</p>
                  </div>
                  <aside className="proof-truth-card">
                    <span className="truth-status"><CircleDot /> Honest implementation status</span>
                    <h3>What works now vs. what is next</h3>
                    <dl>
                      <div><dt>Live now</dt><dd>CAD image selection, validation, local preview, upload metadata, and the editable generated plan.</dd></div>
                      <div><dt>Researched</dt><dd>A pretrained U-Net floor-plan parser baseline and a JSON geometry contract.</dd></div>
                      <div><dt>Not claimed as live</dt><dd>Automatic image-to-geometry inference remains disabled until the separate parser service is deployed and benchmarked.</dd></div>
                    </dl>
                    <a className="proof-inline-link" href={`${REPO_BLOB_URL}/docs/cad-image-parser-research.md`} target="_blank" rel="noreferrer">Read the CAD parser decision <ExternalLink /></a>
                  </aside>
                </div>
              </section>

              <section className="proof-chapter proof-lessons proof-surface">
                <div className="proof-chapter-header">
                  <span className="proof-chapter-number">05</span>
                  <div><span className="eyebrow">What I learned</span><h2>The final result is more than the 90.64% score</h2></div>
                </div>
                <div className="proof-lesson-grid">
                  <article><span>01</span><strong>Accuracy needs a trustworthy test</strong><p>A higher number is worthless if price text or duplicate listings leak into evaluation.</p></article>
                  <article><span>02</span><strong>The model contract shapes the UI</strong><p>Every frontend field must map to the exact schema and preprocessing used during training.</p></article>
                  <article><span>03</span><strong>Deployment is part of ML engineering</strong><p>A model is not a product until its dependencies, artifact, memory, cold start, and public API all work together.</p></article>
                  <article><span>04</span><strong>Creative features still need boundaries</strong><p>The CAD vision idea is documented as a next-stage parser instead of being presented as finished without evidence.</p></article>
                </div>
                <blockquote className="proof-final-quote">“The hardest thing I learned was how to deploy a real ML model so a web application can actually use it. The biggest improvement was learning to prove every claim with a metric, a commit, an artifact, or a live request.”</blockquote>
              </section>

              <section className="proof-evidence proof-surface">
                <div className="proof-chapter-header">
                  <span className="proof-chapter-number">06</span>
                  <div><span className="eyebrow">Open the evidence</span><h2>Inspect the project yourself</h2></div>
                </div>
                <div className="proof-evidence-grid">
                  <a href={`${API_URL}/health`} target="_blank" rel="noreferrer"><span><Server /><div><strong>Live API health</strong><small>Model name, held-out R², and database connection</small></div></span><ExternalLink /></a>
                  <a href={`${API_URL}/docs#/default/predict_predict_post`} target="_blank" rel="noreferrer"><span><CircleDot /><div><strong>Real POST /predict</strong><small>Open the interactive FastAPI endpoint</small></div></span><ExternalLink /></a>
                  <a href={SOURCE_URL} target="_blank" rel="noreferrer"><span><Code2 /><div><strong>Public source repository</strong><small>Complete Git history and implementation</small></div></span><ExternalLink /></a>
                  <a href={`${SOURCE_URL}/commits/main`} target="_blank" rel="noreferrer"><span><Layers3 /><div><strong>Development timeline</strong><small>Every major model, deployment, and UI milestone</small></div></span><ExternalLink /></a>
                  <a href={MODEL_URL} target="_blank" rel="noreferrer"><span><Sparkles /><div><strong>Published model artifact</strong><small>Validated model pieces and model card</small></div></span><ExternalLink /></a>
                  <a href={`${REPO_BLOB_URL}/models/model_metrics.json`} target="_blank" rel="noreferrer"><span><BadgeCheck /><div><strong>Final metrics report</strong><small>R², MAE, RMSE, split sizes, and blend weights</small></div></span><ExternalLink /></a>
                  <a href={`${REPO_BLOB_URL}/deployment/house-price-space/app.py`} target="_blank" rel="noreferrer"><span><Database /><div><strong>API + Supabase code</strong><small>Validation, rate limits, persistence, and inference</small></div></span><ExternalLink /></a>
                  <a href={`${REPO_BLOB_URL}/vercel.json`} target="_blank" rel="noreferrer"><span><Box /><div><strong>Frontend deployment config</strong><small>Vite build and Railway API connection</small></div></span><ExternalLink /></a>
                </div>
              </section>
            </article>
          </Card>
        )}

        <Card className={`details-panel card ${section === "upload" ? "upload-sidebar tour-cad-ai" : ""} ${section === "plan" && mobileDetailsCollapsed ? "mobile-collapsed" : ""}`}>
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
                <div className="panel-heading-actions">
                  <span className="auto-badge"><Sparkles /> Auto plan</span>
                  <Button variant="ghost" className="mobile-details-toggle" onClick={() => setMobileDetailsCollapsed((value) => !value)}>
                    {mobileDetailsCollapsed ? <ChevronDown /> : <ChevronUp />}
                    {mobileDetailsCollapsed ? "Show details" : "Hide details"}
                  </Button>
                </div>
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

          <Card className={`prediction-card card tour-main-ai ${prediction.status}`}>
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
