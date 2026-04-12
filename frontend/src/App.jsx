import { useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import AerialDashboard from "./pages/AerialDashboard";
import LabPage from "./pages/LabPage";
import "./App.css";

const THEMES = {
  dark: {
    "--bg": "#07101f",
    "--bg2": "#0b1628",
    "--bg3": "#0e1c32",
    "--panel": "rgba(11,22,44,0.96)",
    "--panel2": "rgba(8,16,34,0.92)",
    "--panel3": "rgba(6,12,26,0.85)",
    "--border": "rgba(70,120,210,0.17)",
    "--border-hi": "rgba(70,120,210,0.42)",
    "--border-glow": "rgba(70,120,210,0.12)",
    "--text": "#dce8ff",
    "--text-dim": "rgba(150,185,240,0.58)",
    "--text-sub": "rgba(120,155,210,0.38)",
    "--accent": "#2563eb",
    "--accent-mid": "#3b82f6",
    "--accent-dim": "rgba(37,99,235,0.16)",
    "--accent-glow": "rgba(37,99,235,0.30)",
    "--ok": "#10b981",
    "--ok-bg": "rgba(16,185,129,0.09)",
    "--ok-border": "rgba(16,185,129,0.28)",
    "--ok-glow": "rgba(16,185,129,0.20)",
    "--warn": "#f59e0b",
    "--warn-bg": "rgba(245,158,11,0.09)",
    "--warn-border": "rgba(245,158,11,0.28)",
    "--warn-glow": "rgba(245,158,11,0.20)",
    "--crit": "#ef4444",
    "--crit-bg": "rgba(239,68,68,0.09)",
    "--crit-border": "rgba(239,68,68,0.30)",
    "--crit-glow": "rgba(239,68,68,0.22)",
    "--shadow-sm": "0 1px 4px rgba(0,0,0,0.4)",
    "--shadow-md": "0 4px 20px rgba(0,0,0,0.5), 0 0 0 1px rgba(70,120,210,0.08)",
    "--shadow-lg": "0 8px 32px rgba(0,0,0,0.6), 0 0 0 1px rgba(70,120,210,0.10)",
    "--logo-bg": "linear-gradient(140deg,#1e3a8a,#2563eb)",
    "--grid-color": "rgba(70,120,210,0.04)",
    "--stripe": "#1d4ed8",
  },
  light: {
    "--bg": "#edf1f9",
    "--bg2": "#e4eaf5",
    "--bg3": "#dce4f2",
    "--panel": "rgba(255,255,255,0.95)",
    "--panel2": "rgba(248,251,255,0.94)",
    "--panel3": "rgba(240,246,255,0.90)",
    "--border": "rgba(37,99,235,0.13)",
    "--border-hi": "rgba(37,99,235,0.38)",
    "--border-glow": "rgba(37,99,235,0.08)",
    "--text": "#0f172a",
    "--text-dim": "rgba(15,40,100,0.58)",
    "--text-sub": "rgba(15,40,100,0.38)",
    "--accent": "#1d4ed8",
    "--accent-mid": "#2563eb",
    "--accent-dim": "rgba(29,78,216,0.09)",
    "--accent-glow": "rgba(29,78,216,0.20)",
    "--ok": "#059669",
    "--ok-bg": "rgba(5,150,105,0.07)",
    "--ok-border": "rgba(5,150,105,0.24)",
    "--ok-glow": "rgba(5,150,105,0.15)",
    "--warn": "#b45309",
    "--warn-bg": "rgba(180,83,9,0.07)",
    "--warn-border": "rgba(180,83,9,0.24)",
    "--warn-glow": "rgba(180,83,9,0.15)",
    "--crit": "#dc2626",
    "--crit-bg": "rgba(220,38,38,0.07)",
    "--crit-border": "rgba(220,38,38,0.24)",
    "--crit-glow": "rgba(220,38,38,0.15)",
    "--shadow-sm": "0 1px 4px rgba(0,0,0,0.08)",
    "--shadow-md": "0 4px 16px rgba(0,0,0,0.10), 0 0 0 1px rgba(37,99,235,0.07)",
    "--shadow-lg": "0 8px 28px rgba(0,0,0,0.12), 0 0 0 1px rgba(37,99,235,0.08)",
    "--logo-bg": "linear-gradient(140deg,#1e3a8a,#2563eb)",
    "--grid-color": "rgba(37,99,235,0.035)",
    "--stripe": "#1d4ed8",
  },
};

export default function App() {
  const [isDark, setIsDark] = useState(true);

  useEffect(() => {
    const theme = THEMES[isDark ? "dark" : "light"];
    Object.entries(theme).forEach(([key, value]) => {
      document.documentElement.style.setProperty(key, value);
    });
  }, [isDark]);

  return (
    <div className="sp">
      <header className="hdr">
        <div className="hdr-brand">
          <div className="logo">
            <span>SP</span>
          </div>
          <div>
            <div className="brand-name">Sipark</div>
            <div className="brand-sub">Sistema de Monitoreo · Parqueadero de Motos</div>
          </div>
        </div>

        <nav className="nav-tabs" aria-label="Secciones">
          <NavLink to="/" end className="nav-tab">
            Monitoreo Aereo
          </NavLink>
          <NavLink to="/lab" className="nav-tab">
            Laboratorio
          </NavLink>
        </nav>

        <div className="hdr-spacer" />

        <div className="hdr-right shell-right">
          <button className="theme-btn" onClick={() => setIsDark((d) => !d)}>
            {isDark ? "Modo Claro" : "Modo Oscuro"}
          </button>
        </div>
      </header>

      <main className="shell-main">
        <Routes>
          <Route path="/" element={<AerialDashboard isDark={isDark} />} />
          <Route path="/lab" element={<LabPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      <footer className="footer">
        <span>Sipark - Sistema Institucional de Monitoreo de Parqueadero</span>
        <span className="footer-mono">Dashboard + Laboratorio</span>
      </footer>
    </div>
  );
}
