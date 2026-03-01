import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import axios from "axios";
import "./App.css";

const API = "http://localhost:8000";

/* ─── helpers ─────────────────────────────────────────────── */
const fmtTime = (ts) =>
  ts ? new Date(ts * 1000).toLocaleTimeString("es-CO", { hour12: false }) : "--:--:--";
const fmtDate = (ts) =>
  ts
    ? new Date(ts * 1000).toLocaleDateString("es-CO", {
        day: "2-digit", month: "short", year: "numeric",
      })
    : "---";
const sortZoneIds = (ids) =>
  [...ids].sort((a, b) => {
    const na = parseInt(a.replace(/\D/g, ""), 10) || 0;
    const nb = parseInt(b.replace(/\D/g, ""), 10) || 0;
    return na - nb;
  });
const severityOf = (free) => {
  if (free == null) return "na";
  if (free <= 1) return "critical";
  if (free <= 3) return "warn";
  return "ok";
};

/* ─── themes ──────────────────────────────────────────────── */
const THEMES = {
  dark: {
    "--bg":           "#07101f",
    "--bg2":          "#0b1628",
    "--bg3":          "#0e1c32",
    "--panel":        "rgba(11,22,44,0.96)",
    "--panel2":       "rgba(8,16,34,0.92)",
    "--panel3":       "rgba(6,12,26,0.85)",
    "--border":       "rgba(70,120,210,0.17)",
    "--border-hi":    "rgba(70,120,210,0.42)",
    "--border-glow":  "rgba(70,120,210,0.12)",
    "--text":         "#dce8ff",
    "--text-dim":     "rgba(150,185,240,0.58)",
    "--text-sub":     "rgba(120,155,210,0.38)",
    "--accent":       "#2563eb",
    "--accent-mid":   "#3b82f6",
    "--accent-dim":   "rgba(37,99,235,0.16)",
    "--accent-glow":  "rgba(37,99,235,0.30)",
    "--ok":           "#10b981",
    "--ok-bg":        "rgba(16,185,129,0.09)",
    "--ok-border":    "rgba(16,185,129,0.28)",
    "--ok-glow":      "rgba(16,185,129,0.20)",
    "--warn":         "#f59e0b",
    "--warn-bg":      "rgba(245,158,11,0.09)",
    "--warn-border":  "rgba(245,158,11,0.28)",
    "--warn-glow":    "rgba(245,158,11,0.20)",
    "--crit":         "#ef4444",
    "--crit-bg":      "rgba(239,68,68,0.09)",
    "--crit-border":  "rgba(239,68,68,0.30)",
    "--crit-glow":    "rgba(239,68,68,0.22)",
    "--shadow-sm":    "0 1px 4px rgba(0,0,0,0.4)",
    "--shadow-md":    "0 4px 20px rgba(0,0,0,0.5), 0 0 0 1px rgba(70,120,210,0.08)",
    "--shadow-lg":    "0 8px 32px rgba(0,0,0,0.6), 0 0 0 1px rgba(70,120,210,0.10)",
    "--logo-bg":      "linear-gradient(140deg,#1e3a8a,#2563eb)",
    "--grid-color":   "rgba(70,120,210,0.04)",
    "--stripe":       "#1d4ed8",
  },
  light: {
    "--bg":           "#edf1f9",
    "--bg2":          "#e4eaf5",
    "--bg3":          "#dce4f2",
    "--panel":        "rgba(255,255,255,0.95)",
    "--panel2":       "rgba(248,251,255,0.94)",
    "--panel3":       "rgba(240,246,255,0.90)",
    "--border":       "rgba(37,99,235,0.13)",
    "--border-hi":    "rgba(37,99,235,0.38)",
    "--border-glow":  "rgba(37,99,235,0.08)",
    "--text":         "#0f172a",
    "--text-dim":     "rgba(15,40,100,0.58)",
    "--text-sub":     "rgba(15,40,100,0.38)",
    "--accent":       "#1d4ed8",
    "--accent-mid":   "#2563eb",
    "--accent-dim":   "rgba(29,78,216,0.09)",
    "--accent-glow":  "rgba(29,78,216,0.20)",
    "--ok":           "#059669",
    "--ok-bg":        "rgba(5,150,105,0.07)",
    "--ok-border":    "rgba(5,150,105,0.24)",
    "--ok-glow":      "rgba(5,150,105,0.15)",
    "--warn":         "#b45309",
    "--warn-bg":      "rgba(180,83,9,0.07)",
    "--warn-border":  "rgba(180,83,9,0.24)",
    "--warn-glow":    "rgba(180,83,9,0.15)",
    "--crit":         "#dc2626",
    "--crit-bg":      "rgba(220,38,38,0.07)",
    "--crit-border":  "rgba(220,38,38,0.24)",
    "--crit-glow":    "rgba(220,38,38,0.15)",
    "--shadow-sm":    "0 1px 4px rgba(0,0,0,0.08)",
    "--shadow-md":    "0 4px 16px rgba(0,0,0,0.10), 0 0 0 1px rgba(37,99,235,0.07)",
    "--shadow-lg":    "0 8px 28px rgba(0,0,0,0.12), 0 0 0 1px rgba(37,99,235,0.08)",
    "--logo-bg":      "linear-gradient(140deg,#1e3a8a,#2563eb)",
    "--grid-color":   "rgba(37,99,235,0.035)",
    "--stripe":       "#1d4ed8",
  },
};

export default function App() {
  const [isDark, setIsDark] = useState(true);
  const [data, setData]     = useState(null);
  const [conn, setConn]     = useState({ ok: false });
  const [history, setHist]  = useState([]);
  const [lightbox, setLightbox] = useState(false);
  const [zoom, setZoom]     = useState(1);
  const lastTsRef           = useRef(null);

  /* apply theme */
  useEffect(() => {
    const t = THEMES[isDark ? "dark" : "light"];
    Object.entries(t).forEach(([k, v]) =>
      document.documentElement.style.setProperty(k, v));
  }, [isDark]);

  /* lightbox keyboard + scroll */
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") { setLightbox(false); setZoom(1); }
      if (e.key === "+" || e.key === "=") setZoom(z => Math.min(z + 0.25, 4));
      if (e.key === "-") setZoom(z => Math.max(z - 0.25, 0.5));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const handleWheel = useCallback((e) => {
    e.preventDefault();
    setZoom(z => Math.min(Math.max(z - e.deltaY * 0.001, 0.5), 4));
  }, []);

  /* poll backend */
  useEffect(() => {
    const tick = async () => {
      try {
        const res  = await axios.get(`${API}/api/last`, { timeout: 2500 });
        const last = res.data?.data ?? null;
        setData(last);
        setConn({ ok: true });
        const ts = last?.timestamp ?? null;
        if (ts && ts !== lastTsRef.current) {
          lastTsRef.current = ts;
          setHist((p) => [
            ...p,
            { ts, free: last?.totals?.spaces_free ?? 0, occ: last?.totals?.spaces_occupied ?? 0 },
          ].slice(-30));
        }
      } catch {
        setConn({ ok: false });
      }
    };
    tick();
    const id = setInterval(tick, 1500);
    return () => clearInterval(id);
  }, []);

  const totals     = data?.totals;
  const perZone    = data?.per_zone   || {};
  const detections = data?.detections || [];

  const zones = useMemo(() =>
    sortZoneIds(Object.keys(perZone)).map((id) => ({
      id, count: perZone[id] || 0, occupied: (perZone[id] || 0) > 0,
    })), [perZone]);

  const occupancyPct = useMemo(() => {
    if (!totals?.spaces_total) return 0;
    return Math.round((totals.spaces_occupied / totals.spaces_total) * 100);
  }, [totals]);

  const sev = useMemo(() => severityOf(totals?.spaces_free ?? null), [totals]);

  const lastImageUrl = data?.timestamp
    ? `${API}/api/last-image?ts=${data.timestamp}` : null;

  const stripeLabel = {
    ok:       "✓  Operación Normal",
    warn:     "△  Alerta — Cupos Reducidos",
    critical: "⚠  Estado Crítico",
    na:       "◈  Sistema de Monitoreo",
  }[sev];

  const kpis = [
    { label: "Cupos Libres",  value: totals?.spaces_free          ?? "--", tone: sev    },
    { label: "Ocupados",      value: totals?.spaces_occupied       ?? "--", tone:"accent"},
    { label: "Total Cupos",   value: totals?.spaces_total          ?? "--", tone:""      },
    { label: "Detectadas",    value: totals?.motos_detected        ?? "--", tone:""      },
    { label: "Fuera Zonas",   value: totals?.motos_outside_zones   ?? "--", tone:""      },
    { label: "Ocupación",     value: `${occupancyPct}%`,                    tone: sev    },
  ];

  return (
    <>
      <div className="sp">

        {/* ── HEADER ── */}
        <header className="hdr">
          <div className="hdr-brand">
            <div className="logo"><span>SP</span></div>
            <div>
              <div className="brand-name">Sipark</div>
              <div className="brand-sub">Sistema de Monitoreo · Parqueadero de Motos</div>
            </div>
          </div>

          <div className="kpi-bar">
            {kpis.map((k) => (
              <div className="kpi" key={k.label}>
                <span className="kpi-lbl">{k.label}</span>
                <span className={`kpi-val ${k.tone}`}>{k.value}</span>
              </div>
            ))}
          </div>

          <div className="hdr-right">
            <div className="conn">
              <span className={`led ${conn.ok ? "live" : "dead"}`} />
              {conn.ok ? "En línea" : "Sin conexión"}
            </div>
            <div className="ts-wrap">
              <div className="ts-time">{fmtTime(data?.timestamp)}</div>
              <div className="ts-date">{fmtDate(data?.timestamp)}</div>
            </div>
            <button className="theme-btn" onClick={() => setIsDark(d => !d)}>
              {isDark ? "☀ Modo Claro" : "☽ Modo Oscuro"}
            </button>
          </div>
        </header>

        {/* ── MAIN LAYOUT ── */}
        <div className="layout">

          {/* Zone Map */}
          <div className="panel zone-map-panel">
            <div className="panel-head">
              <span className="panel-title">Mapa de Espacios</span>
              <span className="panel-badge">Actualización 1.5 s</span>
            </div>
            <div className="occ-row">
              <div className="occ-track-wrap">
                <div className="occ-labels">
                  <span>Nivel de Ocupación</span>
                  <span>{occupancyPct}%</span>
                </div>
                <div className="occ-track">
                  <div
                    className={`occ-fill ${sev}`}
                    style={{ width: `${occupancyPct}%` }}
                  />
                </div>
              </div>
              <div className={`occ-pct ${sev}`}>
                {occupancyPct}
                <span style={{ fontSize: 14, opacity: .55 }}>%</span>
              </div>
            </div>
            {zones.length ? (
              <div className="zone-grid">
                {zones.map((z, i) => (
                  <div
                    key={z.id}
                    className={`zone-slot ${z.occupied ? "occ" : "free"}`}
                    style={{ animationDelay: `${.05 + i * .03}s` }}
                  >
                    <div className="zs-top">
                      <span className="zs-id">Zona {z.id}</span>
                      <span className="zs-dot" />
                    </div>
                    <div className="zs-count">{z.count}</div>
                    <div className="zs-lbl">{z.occupied ? "Ocupado" : "Libre"}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="zone-empty">
                Sin datos — inicia el backend y el simulador
              </div>
            )}
          </div>

          {/* Camera Feed */}
          <div className="panel camera-panel">
            <div className="panel-head">
              <span className="panel-title">Feed de Cámara</span>
              <span className="panel-badge">
                {data?.timestamp ? "Señal activa" : "Sin señal"}
              </span>
            </div>
            {lastImageUrl ? (
              <div className="cam-frame">
                <img className="cam-img" src={lastImageUrl} alt="Captura" />
                <div className="cam-hud">
                  <div className="hc tl" /><div className="hc tr" />
                  <div className="hc bl" /><div className="hc br" />
                  <div className="cam-bar">
                    <div className="cam-rec">
                      <span className="cam-rec-dot" />Grabando
                    </div>
                    <div className="cam-ts">{fmtTime(data?.timestamp)}</div>
                  </div>
                </div>
                {/* expand button */}
                <button
                  className="cam-expand-btn"
                  onClick={() => { setLightbox(true); setZoom(1); }}
                  title="Expandir imagen"
                >
                  <svg viewBox="0 0 24 24">
                    <polyline points="15 3 21 3 21 9"/>
                    <polyline points="9 21 3 21 3 15"/>
                    <line x1="21" y1="3" x2="14" y2="10"/>
                    <line x1="3" y1="21" x2="10" y2="14"/>
                  </svg>
                </button>
              </div>
            ) : (
              <div className="no-cam">
                <span className="no-cam-icon">⬡</span>
                <span>Sin señal de video</span>
                <span style={{ fontSize: 10, opacity: .5 }}>
                  /api/last-image no disponible
                </span>
              </div>
            )}
          </div>

          {/* ── LIGHTBOX ── */}
          {lightbox && lastImageUrl && (
            <div
              className="lightbox-overlay"
              onClick={() => { setLightbox(false); setZoom(1); }}
              onWheel={handleWheel}
            >
              <div className="lightbox-inner" onClick={e => e.stopPropagation()}>
                {/* toolbar */}
                <div className="lightbox-toolbar">
                  <span className="lb-info">Feed de Cámara · {fmtTime(data?.timestamp)}</span>
                  <div className="lb-controls">
                    {/* zoom out */}
                    <button className="lb-btn" onClick={() => setZoom(z => Math.max(z - 0.25, 0.5))}>
                      <svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
                    </button>
                    <span className="lb-zoom-label">{Math.round(zoom * 100)}%</span>
                    {/* zoom in */}
                    <button className="lb-btn" onClick={() => setZoom(z => Math.min(z + 0.25, 4))}>
                      <svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>
                    </button>
                    {/* reset */}
                    <button className="lb-btn" onClick={() => setZoom(1)}>
                      <svg viewBox="0 0 24 24"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>
                    </button>
                    {/* close */}
                    <button className="lb-btn" onClick={() => { setLightbox(false); setZoom(1); }}>
                      <svg viewBox="0 0 24 24"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                    </button>
                  </div>
                  <span className="lb-esc-hint">ESC para cerrar · rueda para zoom</span>
                </div>

                {/* image frame */}
                <div className="lightbox-frame">
                  <img
                    className="lightbox-img"
                    src={lastImageUrl}
                    alt="Captura expandida"
                    style={{ transform: `scale(${zoom})` }}
                    draggable={false}
                  />
                  {/* HUD corners */}
                  <div className="lb-hc tl"/><div className="lb-hc tr"/>
                  <div className="lb-hc bl"/><div className="lb-hc br"/>
                  {/* bottom bar */}
                  <div className="lightbox-bar">
                    <div className="lb-rec">
                      <span className="lb-rec-dot"/>Grabando en vivo
                    </div>
                    <div className="lb-ts">{fmtTime(data?.timestamp)}</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Trend Chart */}
          <div className="panel trend-panel">
            <div className="panel-head">
              <span className="panel-title">Ocupación en Tiempo Real</span>
              <span className="panel-badge">{history.length} / 30 pts</span>
            </div>
            <OccupancyWidget
              history={history}
              totals={totals}
              sev={sev}
              occupancyPct={occupancyPct}
              isDark={isDark}
            />
          </div>

          {/* ── SIDEBAR ── */}
          <aside className="side-col">

            {/* Primary metric card */}
            <div className="pm-card">
              <div className={`pm-stripe ${sev}`}>{stripeLabel}</div>
              <div className="pm-body">
                <div className="pm-label">Cupos Disponibles</div>
                <div className={`pm-val ${sev}`}>{totals?.spaces_free ?? "--"}</div>
                <div className="pm-sub">
                  de {totals?.spaces_total ?? "--"} cupos totales
                </div>
              </div>
              <div className="pm-divider" />
              <div className="pm-foot">
                <span className="pm-foot-label">Ocupación actual</span>
                <span className="pm-foot-val">{occupancyPct}%</span>
              </div>
            </div>

            {/* Mini metrics */}
            <div className="panel">
              <div className="panel-head">
                <span className="panel-title">Métricas del Sistema</span>
              </div>
              <div className="mini-grid">
                {[
                  { lbl:"Ocupados",    val: totals?.spaces_occupied    ?? "--", cls:"accent" },
                  { lbl:"Detectadas",  val: totals?.motos_detected     ?? "--", cls:"" },
                  { lbl:"Fuera Zonas", val: totals?.motos_outside_zones ?? "--", cls:"" },
                  { lbl:"Total",       val: totals?.spaces_total        ?? "--", cls:"" },
                ].map((m) => (
                  <div className="mc" key={m.lbl}>
                    <div className="mc-lbl">{m.lbl}</div>
                    <div className={`mc-val ${m.cls}`}>{m.val}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Detections */}
            <div className="panel" style={{ flex: 1 }}>
              <div className="panel-head">
                <span className="panel-title">Detecciones Activas</span>
                <span className="panel-badge">Frame actual</span>
              </div>
              {detections.length ? (
                <div className="det-scroll">
                  {detections.slice(0, 12).map((d, i) => (
                    <div
                      className="det-row"
                      key={i}
                      style={{ animationDelay: `${i * .04}s` }}
                    >
                      <div className="det-left">
                        <span className="det-idx">
                          #{String(i + 1).padStart(2, "0")}
                        </span>
                        <span className="det-coord">
                          {Math.round(d.center[0])}, {Math.round(d.center[1])}
                        </span>
                      </div>
                      <span className={`det-tag ${d.zone ? "in" : "out"}`}>
                        {d.zone ?? "Sin zona"}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="no-dets">Sin detecciones en este frame</div>
              )}
            </div>

            {/* Debug JSON */}
            <div className="panel">
              <details>
                <summary className="dbg-summary">
                  <span>JSON Debug</span>
                  <span className="dbg-chevron" style={{ fontSize: 11 }}>▾</span>
                </summary>
                <pre className="dbg-json">
                  {data ? JSON.stringify(data, null, 2) : "// sin datos"}
                </pre>
              </details>
            </div>

          </aside>
        </div>

        {/* ── FOOTER ── */}
        <footer className="footer">
          <span>Sipark — Sistema Institucional de Monitoreo de Parqueadero</span>
          <span className="footer-mono">Backend: {API}</span>
        </footer>

      </div>
    </>
  );
}

/* ─── OccupancyWidget ─────────────────────────────────────── */
function OccupancyWidget({ history, totals, sev, occupancyPct, isDark }) {
  const free  = totals?.spaces_free      ?? 0;
  const occ   = totals?.spaces_occupied  ?? 0;
  const total = totals?.spaces_total     ?? 1;

  /* donut math — r=54 → circumference≈339.3 */
  const R   = 54;
  const CX  = 70;
  const CY  = 70;
  const circ = 2 * Math.PI * R;
  const dash  = circ * (occupancyPct / 100);
  const gap   = circ - dash;

  const okC   = isDark ? "#10b981" : "#059669";
  const critC = isDark ? "#ef4444" : "#dc2626";
  const warnC = isDark ? "#f59e0b" : "#b45309";
  const strokeColor = sev === "ok" ? okC : sev === "warn" ? warnC : sev === "critical" ? critC : "var(--border)";

  /* sparkline */
  const W = 480, H = 80, P = 12;
  const hasHistory = history.length > 1;
  const maxY = hasHistory
    ? Math.max(...history.map((p) => Math.max(p.free, p.occ)), 1)
    : 1;
  const xTo = (i) => P + (i * (W - P * 2)) / Math.max(history.length - 1, 1);
  const yTo = (v) => H - P - (v * (H - P * 2)) / maxY;
  const pts = (k) => history.map((p, i) => `${xTo(i)},${yTo(p[k])}`).join(" ");
  const area = (k) =>
    `${xTo(0)},${H - P} ` +
    history.map((p, i) => `${xTo(i)},${yTo(p[k])}`).join(" ") +
    ` ${xTo(history.length - 1)},${H - P}`;

  const freePct = total ? Math.round((free / total) * 100) : 0;
  const occPct  = total ? Math.round((occ  / total) * 100) : 0;

  return (
    <div className="occ-widget">

      {/* ── donut + side stats ── */}
      <div className="donut-row">
        <div className="donut-wrap">
          <svg className="donut-svg" viewBox="0 0 140 140">
            <defs>
              <filter id="glow">
                <feGaussianBlur stdDeviation="3" result="blur"/>
                <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
              </filter>
            </defs>
            {/* outer decorative ring */}
            <circle cx={CX} cy={CY} r={R + 12} className={`donut-ring ${sev}`} />
            {/* track */}
            <circle cx={CX} cy={CY} r={R} className="donut-track" />
            {/* fill */}
            <circle
              cx={CX} cy={CY} r={R}
              className={`donut-fill ${sev}`}
              strokeDasharray={`${dash} ${gap}`}
              strokeDashoffset={0}
              filter={sev !== "na" ? "url(#glow)" : undefined}
            />
            {/* tick marks */}
            {Array.from({ length: 20 }).map((_, i) => {
              const angle = (i / 20) * 360 - 90;
              const rad   = (angle * Math.PI) / 180;
              const r1    = R + 8, r2 = R + 11;
              return (
                <line
                  key={i}
                  x1={CX + r1 * Math.cos(rad)} y1={CY + r1 * Math.sin(rad)}
                  x2={CX + r2 * Math.cos(rad)} y2={CY + r2 * Math.sin(rad)}
                  stroke="var(--border)" strokeWidth="1.5"
                />
              );
            })}
          </svg>
          <div className="donut-center">
            <span className={`donut-pct ${sev}`}>{occupancyPct}<span style={{fontSize:14}}>%</span></span>
            <span className="donut-lbl">Ocupado</span>
          </div>
        </div>

        <div className="donut-stats">
          {/* free */}
          <div className="ds-item">
            <div className="ds-label">Cupos Libres</div>
            <div className="ds-row">
              <span className="ds-val ok">{free ?? "--"}</span>
              <span className="ds-unit">/ {total}</span>
            </div>
            <div className="ds-bar">
              <div className="ds-bar-fill ok" style={{ width: `${freePct}%` }} />
            </div>
          </div>

          {/* occupied */}
          <div className="ds-item">
            <div className="ds-label">Cupos Ocupados</div>
            <div className="ds-row">
              <span className="ds-val crit">{occ ?? "--"}</span>
              <span className="ds-unit">/ {total}</span>
            </div>
            <div className="ds-bar">
              <div className="ds-bar-fill crit" style={{ width: `${occPct}%` }} />
            </div>
          </div>
        </div>
      </div>

      <div className="w-divider" />

      {/* ── sparkline history ── */}
      <div className="spark-section">
        <div className="spark-head">
          <span className="spark-title">Historial — Últimas {history.length} lecturas</span>
          <span className="spark-pts">{history.length}/30 pts</span>
        </div>

        {hasHistory ? (
          <>
            <svg viewBox={`0 0 ${W} ${H}`} className="spark-svg">
              <defs>
                <linearGradient id="sgF" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"   stopColor={okC}   stopOpacity=".35" />
                  <stop offset="100%" stopColor={okC}   stopOpacity="0"   />
                </linearGradient>
                <linearGradient id="sgO" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"   stopColor={critC} stopOpacity=".25" />
                  <stop offset="100%" stopColor={critC} stopOpacity="0"   />
                </linearGradient>
              </defs>

              {/* grid lines */}
              {[0.33, 0.66].map((t, i) => {
                const y = P + t * (H - P * 2);
                return <line key={i} x1={P} x2={W-P} y1={y} y2={y} className="spark-grid" />;
              })}

              {/* areas */}
              <polygon points={area("occ")}  fill="url(#sgO)" className="spark-area-occ" />
              <polygon points={area("free")} fill="url(#sgF)" className="spark-area-free" />

              {/* lines */}
              <polyline points={pts("occ")}  className="spark-occ"  />
              <polyline points={pts("free")} className="spark-free" />

              {/* latest dots */}
              {(() => {
                const last = history[history.length - 1];
                const lx   = xTo(history.length - 1);
                return (
                  <>
                    <circle cx={lx} cy={yTo(last.free)} r="4" fill={okC}   />
                    <circle cx={lx} cy={yTo(last.occ)}  r="4" fill={critC} />
                    {/* value labels at end */}
                    <text x={lx + 7} y={yTo(last.free) + 4}
                      fontFamily="JetBrains Mono,monospace" fontSize="9"
                      fill={okC} opacity=".9">{last.free}</text>
                    <text x={lx + 7} y={yTo(last.occ) + 4}
                      fontFamily="JetBrains Mono,monospace" fontSize="9"
                      fill={critC} opacity=".9">{last.occ}</text>
                  </>
                );
              })()}
            </svg>

            <div className="spark-legend">
              <div className="s-leg">
                <span className="s-leg-dot free" />Libres
              </div>
              <div className="s-leg">
                <span className="s-leg-dot occ" />Ocupados
              </div>
            </div>
          </>
        ) : (
          <div className="trend-empty">Acumulando datos…</div>
        )}
      </div>
    </div>
  );
}
