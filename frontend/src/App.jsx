import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import axios from "axios";

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

/* ─── CSS ─────────────────────────────────────────────────── */
const CSS = `
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}

body{
  font-family:'DM Sans',sans-serif;
  background-color:var(--bg);
  color:var(--text);
  background-image:
    linear-gradient(var(--grid-color) 1px,transparent 1px),
    linear-gradient(90deg,var(--grid-color) 1px,transparent 1px);
  background-size:28px 28px;
  transition:background-color .4s,color .4s;
  -webkit-font-smoothing:antialiased;
}

/* ── page wrap ── */
.sp{
  max-width:1860px;
  margin:0 auto;
  padding:10px 14px 14px;
  min-height:100vh;
  display:flex;
  flex-direction:column;
  gap:10px;
}

/* ── keyframes ── */
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.42}}
@keyframes ledPulse{0%,100%{opacity:1}50%{opacity:.30}}
@keyframes critPulse{0%,100%{box-shadow:0 0 0 0 var(--crit-glow)}60%{box-shadow:0 0 0 6px transparent}}
@keyframes slideIn{from{opacity:0;transform:translateX(6px)}to{opacity:1;transform:translateX(0)}}
@keyframes barGrow{from{width:0}to{width:var(--w)}}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── HEADER ── */
.hdr{
  display:grid;
  grid-template-columns:auto 1fr auto;
  align-items:stretch;
  background:var(--panel);
  border:1px solid var(--border);
  border-top:3px solid var(--stripe);
  border-radius:8px;
  box-shadow:var(--shadow-lg);
  overflow:hidden;
  transition:background .4s,border-color .4s;
  animation:fadeUp .4s ease both;
}

/* brand */
.hdr-brand{
  display:flex;align-items:center;gap:13px;
  padding:12px 20px 12px 14px;
  border-right:1px solid var(--border);
  background:var(--panel2);
  transition:background .4s;
}
.logo{
  width:44px;height:44px;border-radius:9px;
  background:var(--logo-bg);
  display:grid;place-items:center;
  box-shadow:0 4px 16px var(--accent-glow);
  flex-shrink:0;
  position:relative;
  overflow:hidden;
}
.logo::after{
  content:'';position:absolute;inset:0;
  background:linear-gradient(135deg,rgba(255,255,255,.18) 0%,transparent 60%);
  border-radius:inherit;
}
.logo span{
  font-family:'DM Sans',sans-serif;
  font-size:14px;font-weight:800;
  color:#fff;letter-spacing:.5px;
  position:relative;z-index:1;
}
.brand-name{
  font-family:'DM Sans',sans-serif;
  font-size:18px;font-weight:800;
  letter-spacing:2.5px;text-transform:uppercase;
  color:var(--text);line-height:1;
}
.brand-sub{
  font-family:'JetBrains Mono',monospace;
  font-size:8.5px;color:var(--text-dim);
  letter-spacing:1.2px;margin-top:5px;text-transform:uppercase;
}

/* kpi bar */
.kpi-bar{
  display:flex;align-items:stretch;flex:1;overflow:hidden;
}
.kpi{
  display:flex;flex-direction:column;justify-content:center;align-items:center;
  flex:1;padding:8px 4px;border-right:1px solid var(--border);
  cursor:default;transition:background .2s;position:relative;
}
.kpi:hover{background:var(--accent-dim);}
.kpi-lbl{
  font-family:'DM Sans',sans-serif;
  font-size:9.5px;font-weight:600;
  color:var(--text-dim);text-transform:uppercase;
  letter-spacing:.4px;white-space:nowrap;
}
.kpi-val{
  font-family:'JetBrains Mono',monospace;
  font-size:21px;font-weight:700;
  line-height:1.15;margin-top:1px;
  color:var(--text);
  transition:color .4s;
}
.kpi-val.ok{color:var(--ok)}
.kpi-val.warn{color:var(--warn)}
.kpi-val.critical{color:var(--crit);animation:blink 1s ease-in-out infinite}
.kpi-val.accent{color:var(--accent-mid)}

/* header right */
.hdr-right{
  display:flex;flex-direction:column;justify-content:center;gap:7px;
  padding:10px 14px;border-left:1px solid var(--border);
  min-width:168px;background:var(--panel2);
  transition:background .4s;
}
.conn{
  display:flex;align-items:center;gap:8px;
  font-family:'DM Sans',sans-serif;
  font-size:12px;font-weight:700;color:var(--text);
  letter-spacing:.3px;
}
.led{
  width:9px;height:9px;border-radius:50%;flex-shrink:0;
  transition:background .3s,box-shadow .3s;
}
.led.live{
  background:var(--ok);
  box-shadow:0 0 0 3px var(--ok-bg);
  animation:ledPulse 2s ease-in-out infinite;
}
.led.dead{
  background:var(--crit);
  box-shadow:0 0 0 3px var(--crit-bg);
  animation:ledPulse .45s ease-in-out infinite;
}

.ts-wrap{}
.ts-time{
  font-family:'JetBrains Mono',monospace;
  font-size:19px;font-weight:700;
  color:var(--text);letter-spacing:1px;line-height:1;
}
.ts-date{
  font-family:'JetBrains Mono',monospace;
  font-size:9px;color:var(--text-sub);
  letter-spacing:.8px;margin-top:3px;
}

.theme-btn{
  font-family:'DM Sans',sans-serif;
  font-size:11px;font-weight:600;
  padding:5px 11px;
  border:1px solid var(--border);border-radius:5px;
  background:var(--panel3);color:var(--text-dim);
  cursor:pointer;transition:all .2s;letter-spacing:.2px;
  align-self:flex-start;
}
.theme-btn:hover{
  background:var(--accent-dim);
  border-color:var(--border-hi);
  color:var(--accent);
}

/* ── LAYOUT ── */
.layout{
  display:grid;
  grid-template-columns:1fr 1fr 292px;
  grid-template-rows:auto auto;
  gap:10px;flex:1;align-items:start;
}

/* ── PANEL ── */
.panel{
  background:var(--panel);
  border:1px solid var(--border);
  border-radius:8px;
  box-shadow:var(--shadow-md);
  overflow:hidden;
  transition:background .4s,border-color .4s,box-shadow .3s;
  animation:fadeUp .45s ease both;
}
.panel:hover{box-shadow:var(--shadow-lg);}

.panel-head{
  display:flex;align-items:center;justify-content:space-between;
  padding:9px 14px;border-bottom:1px solid var(--border);
  background:var(--panel2);transition:background .4s;
}
.panel-title{
  font-family:'DM Sans',sans-serif;
  font-size:11px;font-weight:700;
  letter-spacing:.5px;color:var(--accent);
  text-transform:uppercase;
}
.panel-badge{
  font-family:'JetBrains Mono',monospace;
  font-size:8.5px;letter-spacing:.8px;
  color:var(--text-sub);
  border:1px solid var(--border);
  padding:2px 8px;border-radius:4px;
  background:var(--panel);
}

/* ── ZONE MAP ── */
.zone-map-panel{grid-column:1/3;animation-delay:.05s}

/* occupancy bar */
.occ-row{
  display:flex;align-items:center;gap:16px;
  padding:10px 14px;border-bottom:1px solid var(--border);
}
.occ-track-wrap{flex:1}
.occ-labels{
  display:flex;justify-content:space-between;
  font-family:'JetBrains Mono',monospace;
  font-size:8.5px;color:var(--text-dim);
  letter-spacing:.8px;text-transform:uppercase;margin-bottom:5px;
}
.occ-track{
  height:8px;background:var(--panel2);
  border:1px solid var(--border);border-radius:3px;overflow:hidden;
}
.occ-fill{
  height:100%;border-radius:3px;
  transition:width .8s cubic-bezier(.4,0,.2,1);
  position:relative;
}
.occ-fill::after{
  content:'';position:absolute;right:0;top:0;bottom:0;
  width:24px;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.18));
}
.occ-fill.ok{background:linear-gradient(90deg,var(--ok-bg),var(--ok))}
.occ-fill.warn{background:linear-gradient(90deg,var(--warn-bg),var(--warn))}
.occ-fill.critical{background:linear-gradient(90deg,var(--crit-bg),var(--crit))}
.occ-fill.na{background:var(--border)}

.occ-pct{
  font-family:'JetBrains Mono',monospace;
  font-size:28px;font-weight:700;
  min-width:70px;text-align:right;line-height:1;
  transition:color .4s;
}
.occ-pct.ok{color:var(--ok)}
.occ-pct.warn{color:var(--warn)}
.occ-pct.critical{color:var(--crit)}
.occ-pct.na{color:var(--text-dim)}

/* zone grid */
.zone-grid{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(122px,1fr));
  gap:8px;padding:12px;
}
.zone-slot{
  border:1px solid;border-radius:7px;
  padding:11px 12px;
  position:relative;overflow:hidden;
  transition:border-color .25s,box-shadow .25s,transform .2s;
  cursor:default;
  animation:fadeUp .4s ease both;
}
.zone-slot::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  opacity:0;transition:opacity .25s;
}
.zone-slot:hover{transform:translateY(-1px);}
.zone-slot:hover::before{opacity:1}

.zone-slot.free{background:var(--ok-bg);border-color:var(--ok-border)}
.zone-slot.free::before{background:var(--ok)}
.zone-slot.free:hover{box-shadow:0 6px 20px var(--ok-glow);border-color:var(--ok)}

.zone-slot.occ{background:var(--crit-bg);border-color:var(--crit-border)}
.zone-slot.occ::before{background:var(--crit)}
.zone-slot.occ:hover{box-shadow:0 6px 20px var(--crit-glow);border-color:var(--crit)}

.zs-top{display:flex;justify-content:space-between;align-items:center}
.zs-id{
  font-family:'DM Sans',sans-serif;
  font-size:9.5px;font-weight:700;
  letter-spacing:.5px;color:var(--text-dim);text-transform:uppercase;
}
.zs-dot{width:6px;height:6px;border-radius:50%;transition:box-shadow .25s}
.zone-slot.free .zs-dot{background:var(--ok);box-shadow:0 0 5px var(--ok-glow)}
.zone-slot.occ  .zs-dot{background:var(--crit);box-shadow:0 0 5px var(--crit-glow)}

.zs-count{
  font-family:'JetBrains Mono',monospace;
  font-size:32px;font-weight:700;
  line-height:1;margin-top:8px;
  transition:color .3s;
}
.zone-slot.free .zs-count{color:var(--ok)}
.zone-slot.occ  .zs-count{color:var(--crit)}

.zs-lbl{
  font-family:'DM Sans',sans-serif;
  font-size:9px;font-weight:700;
  letter-spacing:.8px;text-transform:uppercase;
  margin-top:4px;color:var(--text-dim);
}
.zone-empty{
  margin:12px;padding:28px;text-align:center;
  font-family:'JetBrains Mono',monospace;
  font-size:11px;color:var(--text-sub);letter-spacing:1px;
  border:1px dashed var(--border);border-radius:6px;
}

/* ── CAMERA ── */
.camera-panel{grid-column:1;grid-row:2;animation-delay:.1s}
.cam-frame{
  position:relative;background:#000;
  aspect-ratio:16/9;overflow:hidden;
}
.cam-img{width:100%;height:100%;object-fit:cover;display:block;}
.cam-hud{position:absolute;inset:0;pointer-events:none}

.hc{
  position:absolute;width:16px;height:16px;
  border-color:rgba(59,130,246,.80);border-style:solid;
  transition:border-color .3s;
}
.cam-frame:hover .hc{border-color:rgba(59,130,246,1)}
.hc.tl{top:8px;left:8px;border-width:2px 0 0 2px;border-radius:2px 0 0 0}
.hc.tr{top:8px;right:8px;border-width:2px 2px 0 0;border-radius:0 2px 0 0}
.hc.bl{bottom:8px;left:8px;border-width:0 0 2px 2px;border-radius:0 0 0 2px}
.hc.br{bottom:8px;right:8px;border-width:0 2px 2px 0;border-radius:0 0 2px 0}

.cam-bar{
  position:absolute;bottom:0;left:0;right:0;
  padding:8px 10px;
  background:linear-gradient(transparent,rgba(0,0,0,.65));
  display:flex;align-items:center;justify-content:space-between;
}
.cam-rec{
  display:flex;align-items:center;gap:6px;
  font-family:'DM Sans',sans-serif;
  font-size:10px;font-weight:600;
  color:rgba(255,255,255,.88);letter-spacing:.8px;text-transform:uppercase;
}
.cam-rec-dot{
  width:7px;height:7px;border-radius:50%;
  background:#ef4444;animation:ledPulse .8s ease-in-out infinite;
}
.cam-ts{
  font-family:'JetBrains Mono',monospace;
  font-size:10px;color:rgba(255,255,255,.60);letter-spacing:.5px;
}

.no-cam{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:8px;aspect-ratio:16/9;
  background:var(--panel2);
  border:1px dashed var(--border);
  border-radius:6px;margin:12px;
  font-family:'DM Sans',sans-serif;
  font-size:12px;font-weight:500;color:var(--text-sub);
}
.no-cam-icon{font-size:28px;opacity:.25}

/* expand button inside cam-frame */
.cam-expand-btn{
  position:absolute;top:10px;right:10px;
  width:32px;height:32px;border-radius:6px;
  background:rgba(0,0,0,.55);
  border:1px solid rgba(255,255,255,.20);
  display:grid;place-items:center;
  cursor:pointer;
  opacity:0;
  transition:opacity .2s,background .2s,border-color .2s;
  z-index:2;
  backdrop-filter:blur(4px);
}
.cam-frame:hover .cam-expand-btn{opacity:1}
.cam-expand-btn:hover{background:rgba(37,99,235,.65);border-color:rgba(59,130,246,.6)}
.cam-expand-btn svg{width:14px;height:14px;stroke:#fff;fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}

/* ── LIGHTBOX ── */
.lightbox-overlay{
  position:fixed;inset:0;z-index:9999;
  background:rgba(0,0,0,.88);
  backdrop-filter:blur(8px);
  display:flex;align-items:center;justify-content:center;
  animation:lbFadeIn .2s ease;
  cursor:zoom-out;
}
@keyframes lbFadeIn{from{opacity:0}to{opacity:1}}

.lightbox-inner{
  position:relative;
  display:flex;flex-direction:column;
  align-items:center;gap:0;
  max-width:92vw;max-height:92vh;
  cursor:default;
  animation:lbScaleIn .22s cubic-bezier(.34,1.56,.64,1);
}
@keyframes lbScaleIn{from{opacity:0;transform:scale(.93)}to{opacity:1;transform:scale(1)}}

.lightbox-frame{
  position:relative;
  border:1px solid rgba(59,130,246,.40);
  border-radius:8px;
  overflow:hidden;
  box-shadow:0 24px 80px rgba(0,0,0,.7),0 0 0 1px rgba(59,130,246,.15);
}
.lightbox-img{
  display:block;
  max-width:90vw;max-height:82vh;
  width:auto;height:auto;
  object-fit:contain;
  transition:transform .15s ease;
  transform-origin:center center;
  user-select:none;
  -webkit-user-drag:none;
}

/* lightbox HUD corners */
.lb-hc{
  position:absolute;width:20px;height:20px;
  border-color:rgba(59,130,246,.9);border-style:solid;
  pointer-events:none;
}
.lb-hc.tl{top:10px;left:10px;border-width:2px 0 0 2px;border-radius:2px 0 0 0}
.lb-hc.tr{top:10px;right:10px;border-width:2px 2px 0 0;border-radius:0 2px 0 0}
.lb-hc.bl{bottom:10px;left:10px;border-width:0 0 2px 2px;border-radius:0 0 0 2px}
.lb-hc.br{bottom:10px;right:10px;border-width:0 2px 2px 0;border-radius:0 0 2px 0}

.lightbox-bar{
  position:absolute;bottom:0;left:0;right:0;
  padding:10px 14px;
  background:linear-gradient(transparent,rgba(0,0,0,.72));
  display:flex;align-items:center;justify-content:space-between;
  pointer-events:none;
}
.lb-rec{
  display:flex;align-items:center;gap:6px;
  font-family:'DM Sans',sans-serif;font-size:11px;font-weight:600;
  color:rgba(255,255,255,.88);letter-spacing:.8px;text-transform:uppercase;
}
.lb-rec-dot{
  width:7px;height:7px;border-radius:50%;
  background:#ef4444;animation:ledPulse .8s ease-in-out infinite;
}
.lb-ts{
  font-family:'JetBrains Mono',monospace;
  font-size:11px;color:rgba(255,255,255,.60);letter-spacing:.5px;
}

/* top toolbar */
.lightbox-toolbar{
  display:flex;align-items:center;justify-content:space-between;
  width:100%;padding:0 4px 8px;
}
.lb-info{
  font-family:'JetBrains Mono',monospace;
  font-size:10px;color:rgba(255,255,255,.45);
  letter-spacing:1px;text-transform:uppercase;
}
.lb-controls{display:flex;gap:6px}
.lb-btn{
  display:grid;place-items:center;
  width:32px;height:32px;border-radius:6px;
  background:rgba(255,255,255,.08);
  border:1px solid rgba(255,255,255,.14);
  cursor:pointer;transition:all .18s;color:#fff;
}
.lb-btn:hover{background:rgba(37,99,235,.5);border-color:rgba(59,130,246,.6)}
.lb-btn svg{width:14px;height:14px;stroke:currentColor;fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.lb-zoom-label{
  font-family:'JetBrains Mono',monospace;
  font-size:10px;color:rgba(255,255,255,.5);
  min-width:36px;text-align:center;letter-spacing:.5px;
}
.lb-esc-hint{
  font-family:'JetBrains Mono',monospace;
  font-size:9px;color:rgba(255,255,255,.28);
  letter-spacing:1px;
}

/* ── OCCUPANCY WIDGET ── */
.trend-panel{grid-column:2;grid-row:2;animation-delay:.15s}
.occ-widget{padding:14px 14px 12px;display:flex;flex-direction:column;gap:14px}

/* donut + center stats */
.donut-row{display:flex;align-items:center;gap:18px}
.donut-wrap{position:relative;flex-shrink:0;width:140px;height:140px}
.donut-svg{width:100%;height:100%;transform:rotate(-90deg)}
.donut-track{fill:none;stroke:var(--border);stroke-width:14}
.donut-fill{
  fill:none;stroke-width:14;stroke-linecap:round;
  transition:stroke-dashoffset .9s cubic-bezier(.4,0,.2,1),stroke .4s;
}
.donut-fill.ok{stroke:var(--ok)}
.donut-fill.warn{stroke:var(--warn)}
.donut-fill.critical{stroke:var(--crit)}
.donut-fill.na{stroke:var(--border)}
/* inner ring */
.donut-ring{fill:none;stroke-width:2;opacity:.35}
.donut-ring.ok{stroke:var(--ok)}
.donut-ring.warn{stroke:var(--warn)}
.donut-ring.critical{stroke:var(--crit)}

.donut-center{
  position:absolute;inset:0;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  pointer-events:none;
}
.donut-pct{
  font-family:'JetBrains Mono',monospace;
  font-size:28px;font-weight:700;line-height:1;
  transition:color .4s;
}
.donut-pct.ok{color:var(--ok)}
.donut-pct.warn{color:var(--warn)}
.donut-pct.critical{color:var(--crit);animation:blink 1s ease-in-out infinite}
.donut-pct.na{color:var(--text-dim)}
.donut-lbl{
  font-family:'DM Sans',sans-serif;
  font-size:9px;font-weight:700;
  text-transform:uppercase;letter-spacing:.8px;
  color:var(--text-sub);margin-top:3px;
}

/* side stats next to donut */
.donut-stats{display:flex;flex-direction:column;gap:10px;flex:1}
.ds-item{}
.ds-label{
  font-family:'DM Sans',sans-serif;
  font-size:9.5px;font-weight:600;
  color:var(--text-sub);text-transform:uppercase;letter-spacing:.4px;
}
.ds-row{display:flex;align-items:baseline;gap:6px;margin-top:2px}
.ds-val{
  font-family:'JetBrains Mono',monospace;
  font-size:20px;font-weight:700;color:var(--text);line-height:1;
}
.ds-val.ok{color:var(--ok)}
.ds-val.crit{color:var(--crit)}
.ds-unit{
  font-family:'JetBrains Mono',monospace;
  font-size:10px;color:var(--text-sub);
}
/* mini inline bar */
.ds-bar{
  margin-top:4px;height:4px;
  background:var(--panel2);border:1px solid var(--border);
  border-radius:2px;overflow:hidden;
}
.ds-bar-fill{
  height:100%;border-radius:2px;
  transition:width .8s cubic-bezier(.4,0,.2,1);
}
.ds-bar-fill.ok{background:var(--ok)}
.ds-bar-fill.crit{background:var(--crit)}

/* divider */
.w-divider{
  height:1px;background:var(--border);
  margin:0 0 2px;
}

/* sparkline history */
.spark-section{}
.spark-head{
  display:flex;align-items:center;justify-content:space-between;
  margin-bottom:8px;
}
.spark-title{
  font-family:'DM Sans',sans-serif;
  font-size:9.5px;font-weight:700;
  text-transform:uppercase;letter-spacing:.5px;color:var(--text-dim);
}
.spark-pts{
  font-family:'JetBrains Mono',monospace;
  font-size:8.5px;color:var(--text-sub);letter-spacing:.5px;
}
.spark-svg{width:100%;display:block;overflow:visible}
.spark-grid{stroke:var(--border);stroke-width:1}
.spark-free{fill:none;stroke:var(--ok);stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.spark-occ{fill:none;stroke:var(--crit);stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.spark-area-free{opacity:.22}
.spark-area-occ{opacity:.16}

.spark-legend{
  display:flex;gap:14px;margin-top:7px;
  font-family:'DM Sans',sans-serif;font-size:10px;font-weight:500;color:var(--text-dim);
}
.s-leg{display:flex;align-items:center;gap:6px}
.s-leg-dot{width:8px;height:8px;border-radius:50%}
.s-leg-dot.free{background:var(--ok)}
.s-leg-dot.occ{background:var(--crit)}

.trend-empty{
  padding:40px 12px;text-align:center;
  font-family:'JetBrains Mono',monospace;
  font-size:10px;color:var(--text-sub);
  letter-spacing:1px;text-transform:uppercase;
}

/* ── SIDEBAR ── */
.side-col{
  grid-column:3;grid-row:1/3;
  display:flex;flex-direction:column;gap:10px;
  animation:slideIn .45s ease both;animation-delay:.05s;
}

/* primary metric */
.pm-card{
  background:var(--panel);
  border:1px solid var(--border);
  border-radius:8px;
  box-shadow:var(--shadow-md);
  overflow:hidden;
  transition:background .4s,box-shadow .3s;
}
.pm-card:hover{box-shadow:var(--shadow-lg)}
.pm-stripe{
  padding:5px 14px;
  font-family:'DM Sans',sans-serif;
  font-size:10px;font-weight:800;
  letter-spacing:1.2px;text-transform:uppercase;
  color:#fff;
  display:flex;align-items:center;gap:7px;
}
.pm-stripe.ok{background:var(--ok)}
.pm-stripe.warn{background:var(--warn)}
.pm-stripe.critical{background:var(--crit);animation:blink 1s ease-in-out infinite}
.pm-stripe.na{background:var(--accent)}
.pm-body{padding:14px 16px}
.pm-label{
  font-family:'DM Sans',sans-serif;
  font-size:10px;font-weight:600;
  color:var(--text-dim);text-transform:uppercase;letter-spacing:.4px;
}
.pm-val{
  font-family:'JetBrains Mono',monospace;
  font-size:64px;font-weight:700;
  line-height:1;margin-top:2px;
  transition:color .4s;
}
.pm-val.ok{color:var(--ok)}
.pm-val.warn{color:var(--warn)}
.pm-val.critical{color:var(--crit)}
.pm-val.na{color:var(--text-dim)}
.pm-sub{
  font-family:'DM Sans',sans-serif;
  font-size:11px;font-weight:400;color:var(--text-dim);margin-top:6px;
}
/* divider line */
.pm-divider{
  height:1px;background:var(--border);margin:12px 16px;
}
.pm-foot{
  padding:0 16px 14px;
  display:flex;align-items:center;justify-content:space-between;
}
.pm-foot-label{
  font-family:'DM Sans',sans-serif;
  font-size:10px;font-weight:600;color:var(--text-sub);text-transform:uppercase;
  letter-spacing:.3px;
}
.pm-foot-val{
  font-family:'JetBrains Mono',monospace;
  font-size:13px;font-weight:700;color:var(--text);
}

/* mini metrics */
.mini-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:10px 12px}
.mc{
  background:var(--panel2);border:1px solid var(--border);
  border-radius:6px;padding:10px 12px;
  transition:background .4s,border-color .25s,transform .2s;cursor:default;
}
.mc:hover{background:var(--accent-dim);border-color:var(--border-hi);transform:translateY(-1px)}
.mc-lbl{
  font-family:'DM Sans',sans-serif;
  font-size:9.5px;font-weight:600;color:var(--text-sub);
  text-transform:uppercase;letter-spacing:.4px;
}
.mc-val{
  font-family:'JetBrains Mono',monospace;
  font-size:22px;font-weight:700;color:var(--text);margin-top:4px;
  transition:color .3s;
}
.mc-val.accent{color:var(--accent-mid)}
.mc-val.ok{color:var(--ok)}

/* detection list */
.det-scroll{
  max-height:264px;overflow-y:auto;
  padding:8px 12px 12px;
  scrollbar-width:thin;scrollbar-color:var(--border) transparent;
}
.det-scroll::-webkit-scrollbar{width:3px}
.det-scroll::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

.det-row{
  display:flex;align-items:center;justify-content:space-between;
  padding:7px 10px;
  border:1px solid var(--border);border-radius:5px;
  margin-bottom:5px;background:var(--panel2);
  gap:8px;transition:border-color .2s,background .2s;
  animation:fadeUp .3s ease both;
}
.det-row:last-child{margin-bottom:0}
.det-row:hover{border-color:var(--border-hi);background:var(--accent-dim)}
.det-left{display:flex;align-items:center;gap:10px}
.det-idx{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--text-sub);min-width:24px}
.det-coord{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text)}
.det-tag{
  font-family:'DM Sans',sans-serif;
  font-size:9.5px;font-weight:700;letter-spacing:.4px;text-transform:uppercase;
  padding:3px 8px;border-radius:4px;border:1px solid;flex-shrink:0;
}
.det-tag.in{background:var(--ok-bg);color:var(--ok);border-color:var(--ok-border)}
.det-tag.out{background:var(--crit-bg);color:var(--crit);border-color:var(--crit-border)}

.no-dets{
  padding:20px 12px;text-align:center;
  font-family:'DM Sans',sans-serif;
  font-size:12px;color:var(--text-sub);
}

/* debug */
.dbg-summary{
  list-style:none;display:flex;align-items:center;justify-content:space-between;
  padding:9px 14px;
  font-family:'DM Sans',sans-serif;
  font-size:11px;font-weight:600;letter-spacing:.3px;
  color:var(--text-dim);cursor:pointer;user-select:none;transition:color .2s;
}
.dbg-summary::-webkit-details-marker{display:none}
details[open] .dbg-summary{color:var(--accent);border-bottom:1px solid var(--border)}
.dbg-chevron{transition:transform .25s}
details[open] .dbg-chevron{transform:rotate(180deg)}
.dbg-json{
  padding:12px 14px;
  font-family:'JetBrains Mono',monospace;
  font-size:9.5px;line-height:1.65;color:var(--text-dim);
  max-height:180px;overflow-y:auto;background:var(--panel2);
}

/* ── FOOTER ── */
.footer{
  display:flex;justify-content:space-between;align-items:center;
  padding:7px 2px 2px;
  border-top:1px solid var(--border);
  font-family:'DM Sans',sans-serif;
  font-size:11px;font-weight:400;color:var(--text-sub);letter-spacing:.2px;
}
.footer-mono{
  font-family:'JetBrains Mono',monospace;
  font-size:9px;letter-spacing:.8px;color:var(--text-sub);
}

/* ═══════════════════════════════════════════
   RESPONSIVE BREAKPOINTS
   ─────────────────────────────────────────
   ≥1300px  →  3-col desktop (default)
   768–1299px →  tablet: 2-col, sidebar below
   <768px   →  mobile: 1-col stack
═══════════════════════════════════════════ */

/* ── TABLET (768–1299px) ── */
@media (max-width:1299px){
  .layout{
    grid-template-columns:1fr 1fr;
    grid-template-rows:auto auto auto;
  }
  .zone-map-panel{grid-column:1/3;grid-row:1}
  .camera-panel{grid-column:1;grid-row:2}
  .trend-panel{grid-column:2;grid-row:2}
  .side-col{
    grid-column:1/3;grid-row:3;
    flex-direction:row;align-items:flex-start;
    flex-wrap:wrap;gap:10px;
  }
  /* on tablet: sidebar items share space in a row */
  .pm-card{flex:0 0 220px}
  .side-col>.panel:nth-child(2){flex:1;min-width:220px}
  .side-col>.panel:nth-child(3){flex:1;min-width:260px}
  .side-col>details{flex:1;min-width:200px}
  .det-scroll{max-height:180px}

  /* collapse KPI bar on smaller tablets */
  .kpi{padding:8px 2px}
  .kpi-lbl{font-size:8.5px}
  .kpi-val{font-size:18px}

  /* header adjustments */
  .hdr{grid-template-columns:auto 1fr auto}
}

/* ── narrow tablet / large phone ── */
@media (max-width:900px){
  .hdr{
    grid-template-columns:1fr;
    grid-template-rows:auto auto auto;
  }
  .hdr-brand{border-right:none;border-bottom:1px solid var(--border)}
  .kpi-bar{
    border-bottom:1px solid var(--border);
    overflow-x:auto;
    -webkit-overflow-scrolling:touch;
  }
  .kpi{min-width:90px}
  .hdr-right{border-left:none;flex-direction:row;align-items:center;flex-wrap:wrap;gap:12px}

  .layout{grid-template-columns:1fr}
  .zone-map-panel{grid-column:1}
  .camera-panel{grid-column:1;grid-row:auto}
  .trend-panel{grid-column:1;grid-row:auto}
  .side-col{
    grid-column:1;grid-row:auto;
    flex-direction:column;
  }
  .pm-card{flex:none}
  .det-scroll{max-height:220px}
}

/* ── mobile ── */
@media (max-width:580px){
  .sp{padding:8px}
  .gap-sm{gap:8px}
  .brand-name{font-size:16px}
  .kpi-val{font-size:17px}
  .kpi-lbl{font-size:8px}
  .zone-grid{grid-template-columns:repeat(auto-fill,minmax(105px,1fr));gap:7px}
  .pm-val{font-size:52px}
  .hdr-right{padding:10px 12px}
  .occ-pct{font-size:22px}
  .ts-time{font-size:16px}
}
`;

/* ─── App ─────────────────────────────────────────────────── */
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
      <style>{CSS}</style>
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