import { useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import "./app.css";

const API = "http://localhost:8000";

function formatTime(ts) {
  if (!ts) return "-";
  return new Date(ts * 1000).toLocaleString();
}

function sortZoneIds(ids) {
  return ids.sort((a, b) => {
    const na = parseInt(a.replace(/\D/g, ""), 10) || 0;
    const nb = parseInt(b.replace(/\D/g, ""), 10) || 0;
    return na - nb;
  });
}

function severityFromFree(free) {
  if (free == null) return "na";
  if (free <= 1) return "critical";
  if (free <= 3) return "warn";
  return "ok";
}

export default function App() {
  const [data, setData] = useState(null);
  const [conn, setConn] = useState({ ok: false, text: "Conectando..." });
  const [history, setHistory] = useState([]); // últimos N frames
  const lastTsRef = useRef(null);

  useEffect(() => {
    const t = setInterval(async () => {
      try {
        const res = await axios.get(`${API}/api/last`, { timeout: 2500 });
        const last = res.data?.data || null;
        setData(last);
        setConn({ ok: true, text: last ? "En línea" : "En línea (sin lecturas)" });

        // guardar historial solo si cambia timestamp
        const ts = last?.timestamp ?? null;
        if (ts && ts !== lastTsRef.current) {
          lastTsRef.current = ts;
          setHistory((prev) => {
            const next = [
              ...prev,
              {
                ts,
                free: last?.totals?.spaces_free ?? 0,
                occ: last?.totals?.spaces_occupied ?? 0,
                det: last?.totals?.motos_detected ?? 0,
              },
            ];
            return next.slice(-30); // 30 lecturas
          });
        }
      } catch {
        setConn({ ok: false, text: "Sin conexión" });
      }
    }, 1500);
    return () => clearInterval(t);
  }, []);

  const totals = data?.totals;
  const perZone = data?.per_zone || {};
  const detections = data?.detections || [];

  const zones = useMemo(() => {
    const ids = sortZoneIds(Object.keys(perZone));
    return ids.map((id) => ({
      id,
      count: perZone[id],
      occupied: (perZone[id] || 0) > 0,
    }));
  }, [perZone]);

  const occupancyPct = useMemo(() => {
    if (!totals?.spaces_total) return 0;
    return Math.round((totals.spaces_occupied / totals.spaces_total) * 100);
  }, [totals]);

  const severity = useMemo(() => severityFromFree(totals?.spaces_free ?? null), [totals]);

  // (Opcional) Vista de cámara: requiere endpoint /api/last-image (ver patch abajo)
  const lastImageUrl = data?.timestamp ? `${API}/api/last-image?ts=${data.timestamp}` : null;

  return (
    <div className="sipark">
      <header className="header">
        <div className="brand">
          <div className="logo">S</div>
          <div className="brandText">
            <div className="brandTop">
              <h1>Sipark</h1>
              <span className={`badge ${conn.ok ? "ok" : "off"}`}>
                <span className={`dot ${conn.ok ? "ok" : "off"}`} />
                {conn.text}
              </span>
              {severity !== "na" && (
                <span className={`badge sev ${severity}`}>
                  {severity === "critical" ? "CRÍTICO" : severity === "warn" ? "ALERTA" : "OK"}
                </span>
              )}
            </div>
            <div className="sub">
              Última lectura: <strong>{formatTime(data?.timestamp)}</strong>
              <span className="sep">•</span>
              Ocupación: <strong>{occupancyPct}%</strong>
            </div>
          </div>
        </div>

        <div className="quick">
          <QuickStat label="Libres" value={totals?.spaces_free ?? "-"} tone={severity} />
          <QuickStat label="Ocupados" value={totals?.spaces_occupied ?? "-"} />
          <QuickStat label="Detectadas" value={totals?.motos_detected ?? "-"} />
        </div>
      </header>

      <main className="layout">
        {/* MAIN: Mapa + Cámara + Historial */}
        <section className="mainCol">
          <div className="panel">
            <div className="panelHead">
              <div>
                <h2>Mapa de espacios</h2>
                <p className="muted">Grid “tipo parqueadero”. Ideal para PC.</p>
              </div>

              <div className="progressBox">
                <div className="progressTop">
                  <span className="muted">Ocupación</span>
                  <strong>{occupancyPct}%</strong>
                </div>
                <div className="bar">
                  <div className="fill" style={{ width: `${occupancyPct}%` }} />
                </div>
              </div>
            </div>

            <div className="spaceGrid">
              {zones.length ? (
                zones.map((z) => (
                  <div key={z.id} className={`space ${z.occupied ? "occ" : "free"}`}>
                    <div className="spaceTop">
                      <span className="spaceId">{z.id}</span>
                      <span className={`pill ${z.occupied ? "occ" : "free"}`}>
                        {z.occupied ? "OCUPADO" : "LIBRE"}
                      </span>
                    </div>
                    <div className="spaceCount">{z.count}</div>
                  </div>
                ))
              ) : (
                <div className="empty">Sin datos aún. Enciende backend + simulador.</div>
              )}
            </div>
          </div>

          <div className="split">
            <div className="panel">
              <div className="panelHead">
                <div>
                  <h2>Vista de cámara</h2>
                  <p className="muted">Muestra la última foto recibida (PC se ve brutal).</p>
                </div>
              </div>

              {lastImageUrl ? (
                <div className="cameraBox">
                  <img className="cameraImg" src={lastImageUrl} alt="Última captura" />
                </div>
              ) : (
                <div className="empty">
                  Sin imagen. (Activa el endpoint <code>/api/last-image</code> con el patch de abajo)
                </div>
              )}
            </div>

            <div className="panel">
              <div className="panelHead">
                <div>
                  <h2>Historial (últimas lecturas)</h2>
                  <p className="muted">Tendencia de libres / ocupados.</p>
                </div>
              </div>

              <TrendChart history={history} />
              <div className="trendLegend">
                <span><i className="sw free" /> Libres</span>
                <span><i className="sw occ" /> Ocupados</span>
              </div>
            </div>
          </div>
        </section>

        {/* SIDEBAR: Métricas + Detecciones + Debug */}
        <aside className="sideCol">
          <div className="panel">
            <h2>Métricas</h2>

            <div className="metrics">
              <Metric big label="Cupos libres" value={totals?.spaces_free ?? "-"} tone={severity} />
              <Metric label="Ocupados" value={totals?.spaces_occupied ?? "-"} />
              <Metric label="Totales" value={totals?.spaces_total ?? "-"} />
              <Metric label="Fuera de zonas" value={totals?.motos_outside_zones ?? "-"} />
            </div>
          </div>

          <div className="panel">
            <div className="panelHead">
              <div>
                <h2>Detecciones recientes</h2>
                <p className="muted">Top 12 del último frame</p>
              </div>
            </div>

            <div className="detList">
              {detections.length ? (
                detections.slice(0, 12).map((d, i) => (
                  <div className="detItem" key={i}>
                    <div>
                      <div className="detTitle">Moto #{i + 1}</div>
                      <div className="muted detSub">
                        ({Math.round(d.center[0])}, {Math.round(d.center[1])})
                      </div>
                    </div>
                    <span className={`tag ${d.zone ? "ok" : "off"}`}>
                      {d.zone ?? "SIN ZONA"}
                    </span>
                  </div>
                ))
              ) : (
                <div className="muted">Sin detecciones aún.</div>
              )}
            </div>
          </div>

          <details className="panel details">
            <summary>
              <div>
                <h2>JSON (debug)</h2>
                <p className="muted">Clic para abrir/cerrar</p>
              </div>
            </summary>
            <pre className="json">{data ? JSON.stringify(data, null, 2) : "Sin datos."}</pre>
          </details>
        </aside>
      </main>

      <footer className="footer">
        <span>Sipark · PC/Tablet Control Room</span>
        <span className="muted">Backend: {API}</span>
      </footer>
    </div>
  );
}

function QuickStat({ label, value, tone }) {
  return (
    <div className="q">
      <div className="qk">{label}</div>
      <div className={`qv ${tone || ""}`}>{value}</div>
    </div>
  );
}

function Metric({ label, value, big, tone }) {
  return (
    <div className={`m ${big ? "big" : ""} ${tone ? `tone-${tone}` : ""}`}>
      <div className="ml">{label}</div>
      <div className="mv">{value}</div>
    </div>
  );
}

/** SVG mini chart sin librerías */
function TrendChart({ history }) {
  const w = 520, h = 160, pad = 14;
  if (!history?.length) {
    return <div className="empty">Aún no hay historial.</div>;
  }

  const xs = history.map((_, i) => i);
  const maxY = Math.max(...history.map((p) => Math.max(p.free, p.occ)), 1);

  const xTo = (i) => pad + (i * (w - pad * 2)) / Math.max(xs.length - 1, 1);
  const yTo = (v) => h - pad - (v * (h - pad * 2)) / maxY;

  const line = (key) =>
    history
      .map((p, i) => `${xTo(i)},${yTo(p[key])}`)
      .join(" ");

  return (
    <div className="trend">
      <svg viewBox={`0 0 ${w} ${h}`} className="trendSvg" role="img" aria-label="Trend">
        {/* grid */}
        {[0.25, 0.5, 0.75].map((t, idx) => {
          const y = pad + t * (h - pad * 2);
          return <line key={idx} x1={pad} x2={w - pad} y1={y} y2={y} className="gridLine" />;
        })}

        <polyline points={line("free")} className="lineFree" />
        <polyline points={line("occ")} className="lineOcc" />
      </svg>
    </div>
  );
}
