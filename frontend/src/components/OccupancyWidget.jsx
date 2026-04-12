export default function OccupancyWidget({ history, totals, sev, occupancyPct, isDark }) {
  const free = totals?.spaces_free ?? 0;
  const occ = totals?.spaces_occupied ?? 0;
  const total = totals?.spaces_total ?? 1;

  const R = 54;
  const CX = 70;
  const CY = 70;
  const circ = 2 * Math.PI * R;
  const dash = circ * (occupancyPct / 100);
  const gap = circ - dash;

  const okC = isDark ? "#10b981" : "#059669";
  const critC = isDark ? "#ef4444" : "#dc2626";
  const warnC = isDark ? "#f59e0b" : "#b45309";

  const W = 480;
  const H = 80;
  const P = 12;
  const hasHistory = history.length > 1;
  const maxY = hasHistory ? Math.max(...history.map((p) => Math.max(p.free, p.occ)), 1) : 1;
  const xTo = (i) => P + (i * (W - P * 2)) / Math.max(history.length - 1, 1);
  const yTo = (v) => H - P - (v * (H - P * 2)) / maxY;
  const pts = (k) => history.map((p, i) => `${xTo(i)},${yTo(p[k])}`).join(" ");
  const area = (k) =>
    `${xTo(0)},${H - P} ` +
    history.map((p, i) => `${xTo(i)},${yTo(p[k])}`).join(" ") +
    ` ${xTo(history.length - 1)},${H - P}`;

  const freePct = total ? Math.round((free / total) * 100) : 0;
  const occPct = total ? Math.round((occ / total) * 100) : 0;

  return (
    <div className="occ-widget">
      <div className="donut-row">
        <div className="donut-wrap">
          <svg className="donut-svg" viewBox="0 0 140 140">
            <defs>
              <filter id="glow">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>
            <circle cx={CX} cy={CY} r={R + 12} className={`donut-ring ${sev}`} />
            <circle cx={CX} cy={CY} r={R} className="donut-track" />
            <circle
              cx={CX}
              cy={CY}
              r={R}
              className={`donut-fill ${sev}`}
              strokeDasharray={`${dash} ${gap}`}
              strokeDashoffset={0}
              filter={sev !== "na" ? "url(#glow)" : undefined}
            />
            {Array.from({ length: 20 }).map((_, i) => {
              const angle = (i / 20) * 360 - 90;
              const rad = (angle * Math.PI) / 180;
              const r1 = R + 8;
              const r2 = R + 11;
              return (
                <line
                  key={i}
                  x1={CX + r1 * Math.cos(rad)}
                  y1={CY + r1 * Math.sin(rad)}
                  x2={CX + r2 * Math.cos(rad)}
                  y2={CY + r2 * Math.sin(rad)}
                  stroke="var(--border)"
                  strokeWidth="1.5"
                />
              );
            })}
          </svg>
          <div className="donut-center">
            <span className={`donut-pct ${sev}`}>
              {occupancyPct}
              <span style={{ fontSize: 14 }}>%</span>
            </span>
            <span className="donut-lbl">Ocupado</span>
          </div>
        </div>

        <div className="donut-stats">
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

      <div className="spark-section">
        <div className="spark-head">
          <span className="spark-title">Historial - Ultimas {history.length} lecturas</span>
          <span className="spark-pts">{history.length}/30 pts</span>
        </div>

        {hasHistory ? (
          <>
            <svg viewBox={`0 0 ${W} ${H}`} className="spark-svg">
              <defs>
                <linearGradient id="sgF" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={okC} stopOpacity=".35" />
                  <stop offset="100%" stopColor={okC} stopOpacity="0" />
                </linearGradient>
                <linearGradient id="sgO" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={critC} stopOpacity=".25" />
                  <stop offset="100%" stopColor={critC} stopOpacity="0" />
                </linearGradient>
              </defs>

              {[0.33, 0.66].map((t, i) => {
                const y = P + t * (H - P * 2);
                return <line key={i} x1={P} x2={W - P} y1={y} y2={y} className="spark-grid" />;
              })}

              <polygon points={area("occ")} fill="url(#sgO)" className="spark-area-occ" />
              <polygon points={area("free")} fill="url(#sgF)" className="spark-area-free" />
              <polyline points={pts("occ")} className="spark-occ" />
              <polyline points={pts("free")} className="spark-free" />

              {(() => {
                const last = history[history.length - 1];
                const lx = xTo(history.length - 1);
                return (
                  <>
                    <circle cx={lx} cy={yTo(last.free)} r="4" fill={okC} />
                    <circle cx={lx} cy={yTo(last.occ)} r="4" fill={critC} />
                    <text
                      x={lx + 7}
                      y={yTo(last.free) + 4}
                      fontFamily="JetBrains Mono,monospace"
                      fontSize="9"
                      fill={okC}
                      opacity=".9"
                    >
                      {last.free}
                    </text>
                    <text
                      x={lx + 7}
                      y={yTo(last.occ) + 4}
                      fontFamily="JetBrains Mono,monospace"
                      fontSize="9"
                      fill={critC}
                      opacity=".9"
                    >
                      {last.occ}
                    </text>
                  </>
                );
              })()}
            </svg>

            <div className="spark-legend">
              <div className="s-leg">
                <span className="s-leg-dot free" />
                Libres
              </div>
              <div className="s-leg">
                <span className="s-leg-dot occ" />
                Ocupados
              </div>
            </div>
          </>
        ) : (
          <div className="trend-empty">Acumulando datos...</div>
        )}
      </div>
    </div>
  );
}
