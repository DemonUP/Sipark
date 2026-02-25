import { CriticalBanner } from "./components/CriticalBanner";
import { StatusBadge } from "./components/StatusBadge";
import { SummaryCard } from "./components/SummaryCard";
import { SpaceCard } from "./components/SpaceCard";
import { DetectionsTable } from "./components/DetectionsTable";
import { OccupancyChart } from "./components/OccupancyChart";
import { ImageWithFallback } from "./components/figma/ImageWithFallback";
import {
  Gauge,
  ParkingSquare,
  Car,
  LayoutGrid,
  Camera,
  Clock,
  MapPin,
} from "lucide-react";

const spaces: Array<{ id: string; status: "OCUPADO" | "LIBRE"; count: number }> = [
  { id: "E1", status: "OCUPADO", count: 1 },
  { id: "E2", status: "LIBRE", count: 0 },
  { id: "E3", status: "OCUPADO", count: 1 },
  { id: "E4", status: "OCUPADO", count: 1 },
  { id: "E5", status: "OCUPADO", count: 1 },
  { id: "E6", status: "OCUPADO", count: 1 },
  { id: "E7", status: "OCUPADO", count: 1 },
  { id: "E8", status: "OCUPADO", count: 1 },
];

const detections = [
  { id: "1", label: "Moto #1", coords: "(868, 694)", space: "E8" },
  { id: "2", label: "Moto #2", coords: "(660, 357)", space: "E5" },
  { id: "3", label: "Moto #3", coords: "(204, 700)", space: "E5" },
  { id: "4", label: "Moto #4", coords: "(842, 686)", space: "E7" },
  { id: "5", label: "Moto #5", coords: "(186, 339)", space: "E1" },
  { id: "6", label: "Moto #6", coords: "(993, 341)", space: "E4" },
];

const isCritical = true;

export default function App() {
  return (
    <div className="min-h-screen bg-[#F4F6F8]" style={{ fontFamily: 'Inter, sans-serif' }}>
      {/* ─── Critical Alert Banner ─── */}
      <CriticalBanner visible={isCritical} />

      {/* ─── Institutional Header ─── */}
      <header className="bg-white border-b border-[#E5E7EB]">
        <div className="max-w-[1440px] mx-auto px-8 py-4 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          {/* Left: Logo + Title */}
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-lg bg-[#0F5E9C] flex items-center justify-center shrink-0">
              <span className="text-white" style={{ fontSize: '20px', fontWeight: 800, letterSpacing: '-0.02em' }}>
                USC
              </span>
            </div>
            <div>
              <h1 className="text-[#0F5E9C]" style={{ fontSize: '18px', fontWeight: 700, lineHeight: 1.3, letterSpacing: '-0.01em' }}>
                Sistema Inteligente de Parqueaderos – SIPARK
              </h1>
              <p className="text-[#6B7280]" style={{ fontSize: '13px', lineHeight: 1.4 }}>
                Universidad Santiago de Cali
              </p>
            </div>
          </div>

          {/* Right: Status Badges + Last Reading */}
          <div className="flex items-center gap-3 flex-wrap">
            <StatusBadge label="Sin conexión" variant="neutral" />
            <StatusBadge label="Crítico" variant="critical" />
            <div className="hidden lg:block h-6 w-px bg-[#E5E7EB]" />
            <div className="flex items-center gap-1.5 text-[#6B7280]">
              <Clock className="w-3.5 h-3.5" />
              <span style={{ fontSize: '12px' }}>
                Última lectura:{" "}
                <span className="text-[#1F2937]" style={{ fontWeight: 600 }}>
                  24/2/2026, 23:29:42
                </span>
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* ─── Main Content ─── */}
      <main className="max-w-[1440px] mx-auto px-8 py-8 flex flex-col gap-8">

        {/* ━━━ SECTION 1: Summary Cards ━━━ */}
        <section>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <SummaryCard
              title="Ocupación general"
              value="88%"
              color="#0F5E9C"
              showBar
              barPercent={88}
              barColor="#0F5E9C"
              icon={<Gauge className="w-5 h-5" />}
            />
            <SummaryCard
              title="Cupos libres"
              value={1}
              color="#2E7D32"
              subtitle="Disponibles en tiempo real"
              icon={<ParkingSquare className="w-5 h-5" />}
            />
            <SummaryCard
              title="Cupos ocupados"
              value={7}
              color="#C62828"
              subtitle="Detectadas: 7 · Fuera de zonas: 0"
              icon={<Car className="w-5 h-5" />}
            />
            <SummaryCard
              title="Total de espacios"
              value={8}
              color="#1F2937"
              subtitle="Capacidad total del parqueadero"
              icon={<LayoutGrid className="w-5 h-5" />}
            />
          </div>
        </section>

        {/* ━━━ SECTION 2: Space Distribution ━━━ */}
        <section className="bg-white rounded-lg border border-[#E5E7EB] p-6" style={{ boxShadow: '0 1px 3px rgba(0,0,0,0.06)' }}>
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-5">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-[#E3F2FD] flex items-center justify-center">
                <MapPin className="w-4 h-4 text-[#0F5E9C]" />
              </div>
              <div>
                <h2 className="text-[#1F2937]" style={{ fontSize: '15px', fontWeight: 600 }}>
                  Distribución de espacios
                </h2>
                <p className="text-[#6B7280]" style={{ fontSize: '12px' }}>
                  Grid de espacios E1–E8 con estado en tiempo real
                </p>
              </div>
            </div>
            {/* Legend */}
            <div className="flex items-center gap-5">
              <div className="flex items-center gap-2">
                <span className="w-3 h-[4px] rounded-full bg-[#C62828]" />
                <span className="text-[#6B7280]" style={{ fontSize: '11px', fontWeight: 500 }}>Ocupado</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="w-3 h-[4px] rounded-full bg-[#2E7D32]" />
                <span className="text-[#6B7280]" style={{ fontSize: '11px', fontWeight: 500 }}>Libre</span>
              </div>
            </div>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
            {spaces.map((s) => (
              <SpaceCard key={s.id} id={s.id} status={s.status} count={s.count} />
            ))}
          </div>
        </section>

        {/* ━━━ SECTION 3: Operational — Camera + Detections ━━━ */}
        <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Camera Card — 7 columns */}
          <div className="lg:col-span-7 bg-white rounded-lg border border-[#E5E7EB] overflow-hidden flex flex-col" style={{ boxShadow: '0 1px 3px rgba(0,0,0,0.06)' }}>
            <div className="px-5 py-4 border-b border-[#E5E7EB] flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-[#E3F2FD] flex items-center justify-center">
                <Camera className="w-4 h-4 text-[#0F5E9C]" />
              </div>
              <div>
                <h3 className="text-[#1F2937]" style={{ fontSize: '15px', fontWeight: 600 }}>
                  Última captura del sistema
                </h3>
                <p className="text-[#6B7280]" style={{ fontSize: '12px' }}>
                  Imagen recibida — 24/2/2026, 23:29:42
                </p>
              </div>
            </div>
            <div className="p-4 flex-1 flex items-center justify-center bg-[#F4F6F8]">
              <div className="w-full rounded-md overflow-hidden border border-[#E5E7EB]">
                <ImageWithFallback
                  src="https://images.unsplash.com/photo-1767782554091-1908642a77d5?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtb3RvcmN5Y2xlJTIwcGFya2luZyUyMGxvdCUyMGFlcmlhbCUyMHZpZXd8ZW58MXx8fHwxNzcxOTk0OTg3fDA&ixlib=rb-4.1.0&q=80&w=1080&utm_source=figma&utm_medium=referral"
                  alt="Vista de cámara del parqueadero SIPARK – Universidad Santiago de Cali"
                  className="w-full h-auto object-cover"
                />
              </div>
            </div>
          </div>

          {/* Detections Table — 5 columns */}
          <div className="lg:col-span-5">
            <DetectionsTable detections={detections} />
          </div>
        </section>

        {/* ━━━ SECTION 4: Occupancy Trend Chart ━━━ */}
        <section>
          <OccupancyChart />
        </section>

        {/* ─── Footer ─── */}
        <footer className="text-center py-5 border-t border-[#E5E7EB]">
          <p className="text-[#9CA3AF]" style={{ fontSize: '11px', letterSpacing: '0.02em' }}>
            SIPARK · Sistema Inteligente de Parqueaderos · Universidad Santiago de Cali · 2026
          </p>
        </footer>
      </main>
    </div>
  );
}
