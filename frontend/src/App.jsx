import { useEffect, useMemo, useRef, useState } from 'react'
import axios from 'axios'
import './App.css'

const API = 'http://localhost:8000'

function formatTime(ts) {
  if (!ts) return '-'
  return new Date(ts * 1000).toLocaleString()
}

function sortZoneIds(ids) {
  return ids.sort((a, b) => {
    const na = parseInt(a.replace(/\D/g, ''), 10) || 0
    const nb = parseInt(b.replace(/\D/g, ''), 10) || 0
    return na - nb
  })
}

function severityFromFree(free) {
  if (free == null) return 'na'
  if (free <= 1) return 'critical'
  if (free <= 3) return 'warn'
  return 'ok'
}

function StatusBadge({ label, variant }) {
  return (
    <span className={`status-badge ${variant}`}>
      {variant === 'critical' && <span className="status-dot pulse" />}
      {variant === 'neutral' && <span className="status-dot neutral" />}
      {label}
    </span>
  )
}

function SummaryCard({ title, value, color, subtitle, showBar, barPercent }) {
  return (
    <div className="summary-card">
      <div className="summary-head">
        <span className="summary-title">{title}</span>
      </div>
      <span className="summary-value" style={{ color }}>{value}</span>
      {showBar ? (
        <div className="summary-bar-wrap">
          <div className="summary-bar">
            <div className="summary-bar-fill" style={{ width: `${barPercent}%` }} />
          </div>
          <span className="summary-note">{barPercent}% de capacidad utilizada</span>
        </div>
      ) : (
        <span className="summary-note">{subtitle}</span>
      )}
    </div>
  )
}

function SpaceCard({ id, occupied }) {
  return (
    <div className="space-card">
      <div className="space-stripe" style={{ backgroundColor: occupied ? '#C62828' : '#2E7D32' }} />
      <div className="space-content">
        <span className="space-id">{id}</span>
        <span className={`space-icon ${occupied ? 'occupied' : 'free'}`}>{occupied ? '🏍' : '✓'}</span>
        <span className={`space-state ${occupied ? 'occupied' : 'free'}`}>{occupied ? 'OCUPADO' : 'LIBRE'}</span>
      </div>
    </div>
  )
}

function Chart({ data, maxY }) {
  if (!data.length) return <div className="empty-chart">Sin lecturas aún</div>
  const width = 900
  const height = 260
  const pl = 30
  const pb = 24
  const pw = width - pl - 20
  const ph = height - 20 - pb
  const x = (i) => pl + (i * pw) / Math.max(1, data.length - 1)
  const y = (v) => 20 + ph - (v / Math.max(1, maxY)) * ph
  const path = (k) => data.map((d, i) => `${i ? 'L' : 'M'} ${x(i)} ${y(d[k])}`).join(' ')

  return (
    <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      {[0.25, 0.5, 0.75].map((g) => <line key={g} x1={pl} y1={20 + ph * g} x2={pl + pw} y2={20 + ph * g} className="grid-line" />)}
      <path d={path('ocupados')} className="line-oc" />
      <path d={path('libres')} className="line-free" />
    </svg>
  )
}

export default function App() {
  const [data, setData] = useState(null)
  const [conn, setConn] = useState({ ok: false, text: 'Conectando...' })
  const [history, setHistory] = useState([])
  const lastTsRef = useRef(null)

  useEffect(() => {
    const t = setInterval(async () => {
      try {
        const res = await axios.get(`${API}/api/last`, { timeout: 2500 })
        const last = res.data?.data || null
        setData(last)
        setConn({ ok: true, text: last ? 'En línea' : 'En línea (sin lecturas)' })

        const ts = last?.timestamp ?? null
        if (ts && ts !== lastTsRef.current) {
          lastTsRef.current = ts
          setHistory((prev) => [...prev, { ts, free: last?.totals?.spaces_free ?? 0, occ: last?.totals?.spaces_occupied ?? 0 }].slice(-30))
        }
      } catch {
        setConn({ ok: false, text: 'Sin conexión' })
      }
    }, 1500)
    return () => clearInterval(t)
  }, [])

  const totals = data?.totals
  const perZone = data?.per_zone || {}
  const detections = data?.detections || []

  const zones = useMemo(() => {
    const ids = sortZoneIds(Object.keys(perZone))
    return ids.map((id) => ({ id, count: perZone[id], occupied: (perZone[id] || 0) > 0 }))
  }, [perZone])

  const occupancyPct = useMemo(() => {
    if (!totals?.spaces_total) return 0
    return Math.round((totals.spaces_occupied / totals.spaces_total) * 100)
  }, [totals])

  const severity = useMemo(() => severityFromFree(totals?.spaces_free ?? null), [totals])
  const isCritical = !conn.ok || severity === 'critical'
  const lastImageUrl = data?.timestamp ? `${API}/api/last-image?ts=${data.timestamp}` : null
  const chartData = history.map((h) => ({ libres: h.free, ocupados: h.occ }))

  return (
    <div className="sipark">

      <header className="header">
        <div className="header-inner">
          <div className="brand"><div className="logo">USC</div><div><h1>Sistema Inteligente de Parqueaderos – SIPARK</h1><p>Universidad Santiago de Cali</p></div></div>
          <div className="header-right">
            <StatusBadge label={conn.text} variant={conn.ok ? 'success' : 'neutral'} />
            <StatusBadge label={severity === 'critical' ? 'Crítico' : severity === 'warn' ? 'Alerta' : 'Estable'} variant={severity === 'critical' ? 'critical' : severity === 'warn' ? 'warning' : 'info'} />
            <span className="last-read">Última lectura: <strong>{formatTime(data?.timestamp)}</strong></span>
          </div>
        </div>
      </header>

      <main className="main">
        <section className="summary-grid">
          <SummaryCard title="Ocupación general" value={`${occupancyPct}%`} color="#0F5E9C" showBar barPercent={occupancyPct} />
          <SummaryCard title="Cupos libres" value={totals?.spaces_free ?? '-'} color="#2E7D32" subtitle="Disponibles en tiempo real" />
          <SummaryCard title="Cupos ocupados" value={totals?.spaces_occupied ?? '-'} color="#C62828" subtitle={`Detectadas: ${totals?.motos_detected ?? 0} · Fuera de zonas: ${totals?.motos_outside_zone ?? 0}`} />
          <SummaryCard title="Total de espacios" value={totals?.spaces_total ?? '-'} color="#1F2937" subtitle="Capacidad total del parqueadero" />
        </section>

        <section className="card">
          <div className="section-head"><h2>Distribución de espacios</h2></div>
          <div className="spaces-grid">{zones.map((z) => <SpaceCard key={z.id} id={z.id} occupied={z.occupied} />)}</div>
        </section>

        <section className="ops-grid">
          <div className="card">
            <div className="section-head"><h2>Última captura del sistema</h2><p>Imagen recibida — {formatTime(data?.timestamp)}</p></div>
            <div className="camera-wrap"><img src={lastImageUrl || 'https://images.unsplash.com/photo-1767782554091-1908642a77d5?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&q=80&w=1080'} alt="Cámara" /></div>
          </div>

          <div className="card table-card">
            <div className="section-head"><h2>Detecciones recientes</h2></div>
            <table>
              <thead><tr><th>ID</th><th>VEHÍCULO</th><th>COORDENADAS</th><th>ESPACIO</th></tr></thead>
              <tbody>
                {detections.map((d, i) => (
                  <tr key={i}>
                    <td>{d.id ?? i + 1}</td>
                    <td>{d.label ?? d.class_name ?? 'Moto'}</td>
                    <td><code>{d.coords ?? `(${d.cx ?? 0}, ${d.cy ?? 0})`}</code></td>
                    <td><span className="zone-tag">{d.space ?? d.zone_id ?? '-'}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="card">
          <div className="section-head"><h2>Tendencia de ocupación reciente</h2></div>
          <Chart data={chartData} maxY={Math.max(8, totals?.spaces_total ?? 8)} />
          <div className="legend"><span><i className="occ" />Ocupados</span><span><i className="free" />Libres</span></div>
        </section>

        <footer className="footer">SIPARK · Sistema Inteligente de Parqueaderos · Universidad Santiago de Cali · 2026</footer>
      </main>
    </div>
  )
}
