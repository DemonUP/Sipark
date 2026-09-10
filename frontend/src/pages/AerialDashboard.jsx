import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import OccupancyWidget from "../components/OccupancyWidget";

const VIEWPOINT_LABELS = {
  superior: "Aerea (superior)",
  oblicua: "Oblicua (a nivel)",
  indeterminado: "Sin determinar",
};

const API = "http://localhost:8000";

const fmtTime = (ts) =>
  ts ? new Date(ts * 1000).toLocaleTimeString("es-CO", { hour12: false }) : "--:--:--";

const fmtDate = (ts) =>
  ts
    ? new Date(ts * 1000).toLocaleDateString("es-CO", {
        day: "2-digit",
        month: "short",
        year: "numeric",
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

export default function AerialDashboard({ isDark }) {
  const [data, setData] = useState(null);
  const [conn, setConn] = useState({ ok: false });
  const [history, setHist] = useState([]);
  const [lightbox, setLightbox] = useState(false);
  const [zoom, setZoom] = useState(1);
  const lastTsRef = useRef(null);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "Escape") {
        setLightbox(false);
        setZoom(1);
      }
      if (e.key === "+" || e.key === "=") setZoom((z) => Math.min(z + 0.25, 4));
      if (e.key === "-") setZoom((z) => Math.max(z - 0.25, 0.5));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const handleWheel = useCallback((e) => {
    e.preventDefault();
    setZoom((z) => Math.min(Math.max(z - e.deltaY * 0.001, 0.5), 4));
  }, []);

  useEffect(() => {
    const tick = async () => {
      try {
        const res = await axios.get(`${API}/api/last`, { timeout: 2500 });
        const last = res.data?.data ?? null;
        setData(last);
        setConn({ ok: true });
        const ts = last?.timestamp ?? null;
        if (ts && ts !== lastTsRef.current) {
          lastTsRef.current = ts;
          setHist((p) =>
            [...p, { ts, free: last?.totals?.spaces_free ?? 0, occ: last?.totals?.spaces_occupied ?? 0 }].slice(-30),
          );
        }
      } catch {
        setConn({ ok: false });
      }
    };

    tick();
    const id = setInterval(tick, 1500);
    return () => clearInterval(id);
  }, []);

  const totals = data?.totals;
  const scene = data?.scene;
  const diagnostics = data?.diagnostics;
  // El backend avisa cuando la vista que entrega la camara no corresponde a la
  // que se uso para dibujar las zonas. En ese caso la ocupacion por zona no
  // describe cupos, y mostrarla como un dato firme seria enganoso.
  const zonesApply = diagnostics?.zones_apply !== false;
  const perZone = useMemo(() => data?.per_zone || {}, [data?.per_zone]);
  const detections = data?.detections || [];
  const zones = useMemo(
    () =>
      sortZoneIds(Object.keys(perZone)).map((id) => ({
        id,
        count: perZone[id] || 0,
        occupied: (perZone[id] || 0) > 0,
      })),
    [perZone],
  );
  const occupancyPct = useMemo(() => {
    if (!totals?.spaces_total) return 0;
    return Math.round((totals.spaces_occupied / totals.spaces_total) * 100);
  }, [totals]);
  const sev = useMemo(() => severityOf(totals?.spaces_free ?? null), [totals]);
  const lastImageUrl = data?.timestamp ? `${API}/api/last-image?ts=${data.timestamp}` : null;

  const stripeLabel = {
    ok: "Operacion Normal",
    warn: "Alerta - Cupos Reducidos",
    critical: "Estado Critico",
    na: "Sistema de Monitoreo",
  }[sev];

  // Las motos detectadas se miden en la imagen y se informan siempre; los cupos
  // dependen de que las zonas correspondan a esta camara.
  const zoneValue = (value) => (zonesApply ? value ?? "--" : "n/d");
  const kpis = [
    { label: "Cupos Libres", value: zoneValue(totals?.spaces_free), tone: zonesApply ? sev : "" },
    { label: "Ocupados", value: zoneValue(totals?.spaces_occupied), tone: zonesApply ? "accent" : "" },
    { label: "Total Cupos", value: totals?.spaces_total ?? "--", tone: "" },
    { label: "Detectadas", value: totals?.motos_detected ?? "--", tone: "" },
    { label: "Fuera Zonas", value: zoneValue(totals?.motos_outside_zones), tone: "" },
    { label: "Ocupacion", value: zonesApply ? `${occupancyPct}%` : "n/d", tone: zonesApply ? sev : "" },
  ];

  return (
    <div className="dash-shell">
      <section className="subbar">
        <div className="kpi-bar">
          {kpis.map((k) => (
            <div className="kpi" key={k.label}>
              <span className="kpi-lbl">{k.label}</span>
              <span className={`kpi-val ${k.tone}`}>{k.value}</span>
            </div>
          ))}
        </div>

        <div className="subbar-meta">
          <div className="conn">
            <span className={`led ${conn.ok ? "live" : "dead"}`} />
            {conn.ok ? "En linea" : "Sin conexion"}
          </div>
          <div className="ts-wrap">
            <div className="ts-time">{fmtTime(data?.timestamp)}</div>
            <div className="ts-date">{fmtDate(data?.timestamp)}</div>
          </div>
        </div>
      </section>

      {!zonesApply && (
        <section className="geo-warning">
          <span className="geo-warning-tag">Ocupacion no representativa</span>
          {(diagnostics?.notes ?? []).map((note) => (
            <p key={note}>{note}</p>
          ))}
        </section>
      )}

      {scene && (
        <section className="geo-bar">
          <div className="geo-item">
            <span className="geo-lbl">Vista detectada</span>
            <span className="geo-val">{VIEWPOINT_LABELS[scene.viewpoint] ?? scene.viewpoint}</span>
          </div>
          <div className="geo-item">
            <span className="geo-lbl">Moto tipica</span>
            <span className="geo-val">{scene.median_side_px} px</span>
          </div>
          <div className="geo-item">
            <span className="geo-lbl">Mosaicos</span>
            <span className="geo-val">{scene.tiles_used}</span>
          </div>
          <div className="geo-item">
            <span className="geo-lbl">Proceso</span>
            <span className="geo-val">{diagnostics?.elapsed_ms ?? "--"} ms</span>
          </div>
        </section>
      )}

      <div className="layout">
        <div className="panel zone-map-panel">
          <div className="panel-head">
            <span className="panel-title">Mapa de Espacios</span>
            <span className="panel-badge">Actualizacion 1.5 s</span>
          </div>
          <div className="occ-row">
            <div className="occ-track-wrap">
              <div className="occ-labels">
                <span>Nivel de Ocupacion</span>
                <span>{occupancyPct}%</span>
              </div>
              <div className="occ-track">
                <div className={`occ-fill ${sev}`} style={{ width: `${occupancyPct}%` }} />
              </div>
            </div>
            <div className={`occ-pct ${sev}`}>
              {occupancyPct}
              <span style={{ fontSize: 14, opacity: 0.55 }}>%</span>
            </div>
          </div>
          {zones.length ? (
            <div className="zone-grid">
              {zones.map((z, i) => (
                <div key={z.id} className={`zone-slot ${z.occupied ? "occ" : "free"}`} style={{ animationDelay: `${0.05 + i * 0.03}s` }}>
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
            <div className="zone-empty">Sin datos - inicia el backend y el simulador</div>
          )}
        </div>

        <div className="panel camera-panel">
          <div className="panel-head">
            <span className="panel-title">Feed de Camara</span>
            <span className="panel-badge">{data?.timestamp ? "Senal activa" : "Sin senal"}</span>
          </div>
          {lastImageUrl ? (
            <div className="cam-frame">
              <img className="cam-img" src={lastImageUrl} alt="Captura" />
              <div className="cam-hud">
                <div className="hc tl" />
                <div className="hc tr" />
                <div className="hc bl" />
                <div className="hc br" />
                <div className="cam-bar">
                  <div className="cam-rec">
                    <span className="cam-rec-dot" />
                    Grabando
                  </div>
                  <div className="cam-ts">{fmtTime(data?.timestamp)}</div>
                </div>
              </div>
              <button className="cam-expand-btn" onClick={() => { setLightbox(true); setZoom(1); }} title="Expandir imagen">
                <svg viewBox="0 0 24 24">
                  <polyline points="15 3 21 3 21 9" />
                  <polyline points="9 21 3 21 3 15" />
                  <line x1="21" y1="3" x2="14" y2="10" />
                  <line x1="3" y1="21" x2="10" y2="14" />
                </svg>
              </button>
            </div>
          ) : (
            <div className="no-cam">
              <span className="no-cam-icon">[]</span>
              <span>Sin senal de video</span>
              <span style={{ fontSize: 10, opacity: 0.5 }}>/api/last-image no disponible</span>
            </div>
          )}
        </div>

        {lightbox && lastImageUrl && (
          <div className="lightbox-overlay" onClick={() => { setLightbox(false); setZoom(1); }} onWheel={handleWheel}>
            <div className="lightbox-inner" onClick={(e) => e.stopPropagation()}>
              <div className="lightbox-toolbar">
                <span className="lb-info">Feed de Camara - {fmtTime(data?.timestamp)}</span>
                <div className="lb-controls">
                  <button className="lb-btn" onClick={() => setZoom((z) => Math.max(z - 0.25, 0.5))}>
                    <svg viewBox="0 0 24 24">
                      <circle cx="11" cy="11" r="8" />
                      <line x1="21" y1="21" x2="16.65" y2="16.65" />
                      <line x1="8" y1="11" x2="14" y2="11" />
                    </svg>
                  </button>
                  <span className="lb-zoom-label">{Math.round(zoom * 100)}%</span>
                  <button className="lb-btn" onClick={() => setZoom((z) => Math.min(z + 0.25, 4))}>
                    <svg viewBox="0 0 24 24">
                      <circle cx="11" cy="11" r="8" />
                      <line x1="21" y1="21" x2="16.65" y2="16.65" />
                      <line x1="11" y1="8" x2="11" y2="14" />
                      <line x1="8" y1="11" x2="14" y2="11" />
                    </svg>
                  </button>
                  <button className="lb-btn" onClick={() => setZoom(1)}>
                    <svg viewBox="0 0 24 24">
                      <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                      <path d="M3 3v5h5" />
                    </svg>
                  </button>
                  <button className="lb-btn" onClick={() => { setLightbox(false); setZoom(1); }}>
                    <svg viewBox="0 0 24 24">
                      <line x1="18" y1="6" x2="6" y2="18" />
                      <line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                  </button>
                </div>
                <span className="lb-esc-hint">ESC para cerrar - rueda para zoom</span>
              </div>

              <div className="lightbox-frame">
                <img
                  className="lightbox-img"
                  src={lastImageUrl}
                  alt="Captura expandida"
                  style={{ transform: `scale(${zoom})` }}
                  draggable={false}
                />
                <div className="lb-hc tl" />
                <div className="lb-hc tr" />
                <div className="lb-hc bl" />
                <div className="lb-hc br" />
                <div className="lightbox-bar">
                  <div className="lb-rec">
                    <span className="lb-rec-dot" />
                    Grabando en vivo
                  </div>
                  <div className="lb-ts">{fmtTime(data?.timestamp)}</div>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="panel trend-panel">
          <div className="panel-head">
            <span className="panel-title">Ocupacion en Tiempo Real</span>
            <span className="panel-badge">{history.length} / 30 pts</span>
          </div>
          <OccupancyWidget history={history} totals={totals} sev={sev} occupancyPct={occupancyPct} isDark={isDark} />
        </div>

        <aside className="side-col">
          <div className="pm-card">
            <div className={`pm-stripe ${sev}`}>{stripeLabel}</div>
            <div className="pm-body">
              <div className="pm-label">Cupos Disponibles</div>
              <div className={`pm-val ${sev}`}>{totals?.spaces_free ?? "--"}</div>
              <div className="pm-sub">de {totals?.spaces_total ?? "--"} cupos totales</div>
            </div>
            <div className="pm-divider" />
            <div className="pm-foot">
              <span className="pm-foot-label">Ocupacion actual</span>
              <span className="pm-foot-val">{occupancyPct}%</span>
            </div>
          </div>

          <div className="panel">
            <div className="panel-head">
              <span className="panel-title">Metricas del Sistema</span>
            </div>
            <div className="mini-grid">
              {[
                { lbl: "Ocupados", val: totals?.spaces_occupied ?? "--", cls: "accent" },
                { lbl: "Detectadas", val: totals?.motos_detected ?? "--", cls: "" },
                { lbl: "Fuera Zonas", val: totals?.motos_outside_zones ?? "--", cls: "" },
                { lbl: "Total", val: totals?.spaces_total ?? "--", cls: "" },
              ].map((m) => (
                <div className="mc" key={m.lbl}>
                  <div className="mc-lbl">{m.lbl}</div>
                  <div className={`mc-val ${m.cls}`}>{m.val}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="panel" style={{ flex: 1 }}>
            <div className="panel-head">
              <span className="panel-title">Detecciones Activas</span>
              <span className="panel-badge">Frame actual</span>
            </div>
            {detections.length ? (
              <div className="det-scroll">
                {detections.slice(0, 12).map((d, i) => (
                  <div className="det-row" key={i} style={{ animationDelay: `${i * 0.04}s` }}>
                    <div className="det-left">
                      <span className="det-idx">#{String(i + 1).padStart(2, "0")}</span>
                      <span className="det-coord">
                        {Math.round(d.center[0])}, {Math.round(d.center[1])}
                      </span>
                    </div>
                    <span className={`det-tag ${d.zone ? "in" : "out"}`}>{d.zone ?? "Sin zona"}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="no-dets">Sin detecciones en este frame</div>
            )}
          </div>

          <div className="panel">
            <details>
              <summary className="dbg-summary">
                <span>JSON Debug</span>
                <span className="dbg-chevron" style={{ fontSize: 11 }}>
                  v
                </span>
              </summary>
              <pre className="dbg-json">{data ? JSON.stringify(data, null, 2) : "// sin datos"}</pre>
            </details>
          </div>
        </aside>
      </div>
    </div>
  );
}
