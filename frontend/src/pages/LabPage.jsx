import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import "./LabPage.css";

const API = "http://localhost:8000";

const toneForConfidence = (value) => {
  if (value >= 0.7) return "good";
  if (value >= 0.4) return "warn";
  return "low";
};

const lightingBadge = (value) => {
  if (value === "Dia") return "SUN";
  if (value === "Tarde") return "PM";
  if (value === "Noche") return "MOON";
  return "--";
};

const imageSrcFor = (filename, analysis, conf, minArea) => {
  if (!filename) return null;
  const encoded = encodeURIComponent(filename);
  if (!analysis) return `${API}/api/lab/images/${encoded}`;
  return `${API}/api/lab/annotated/${encoded}?conf=${conf}&min_area=${minArea}`;
};

export default function LabPage() {
  const [images, setImages] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState(null);
  const [conf, setConf] = useState(0.25);
  const [minArea, setMinArea] = useState(5000);

  useEffect(() => {
    const loadImages = async () => {
      const res = await axios.get(`${API}/api/lab/images`);
      const files = Array.isArray(res.data) ? res.data : res.data?.images ?? [];
      setImages(files);
      setSelected((current) => current ?? files[0] ?? null);
    };

    loadImages().catch(() => {
      setImages([]);
    });
  }, []);

  const imageMap = useMemo(() => {
    const map = new Map();
    for (const item of analysis?.images ?? []) map.set(item.filename, item);
    return map;
  }, [analysis]);

  const selectedImage = selected ? imageMap.get(selected) ?? null : null;
  const selectedPreview = selected ? imageSrcFor(selected, analysis, conf, minArea) : null;

  const predominantAngle = useMemo(() => {
    if (!selectedImage?.detections?.length) return "--";
    const counts = selectedImage.detections.reduce((acc, item) => {
      acc[item.angle_class] = (acc[item.angle_class] ?? 0) + 1;
      return acc;
    }, {});
    return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
  }, [selectedImage]);

  const runAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post(`${API}/api/lab/analyze-dataset`, null, {
        params: { conf, min_area: minArea },
        timeout: 300000,
      });
      setAnalysis(res.data);
      const first = res.data?.images?.[0]?.filename;
      setSelected((current) => current ?? first ?? null);
    } catch (err) {
      const msg = err.code === "ECONNABORTED"
        ? "Timeout: el análisis tardó demasiado. Prueba subir la confianza o reinicia el backend."
        : err.response?.data?.detail ?? err.message ?? "Error desconocido al analizar.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const exportReport = async () => {
    setExportError(null);
    setExporting(true);
    try {
      const res = await axios.get(`${API}/api/lab/report`, {
        params: { conf, min_area: minArea },
        responseType: "blob",
        timeout: 300000,
      });
      const blobUrl = URL.createObjectURL(new Blob([res.data]));
      const link = document.createElement("a");
      link.href = blobUrl;
      link.download = "sipark_lab_report.xlsx";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(blobUrl);
    } catch (err) {
      const msg = err.code === "ECONNABORTED"
        ? "Timeout al exportar. Ejecuta el análisis primero."
        : err.response?.data?.detail ?? err.message ?? "Error al exportar.";
      setExportError(msg);
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="lab-page">
      <section className="lab-toolbar panel">
        <div className="panel-head">
          <span className="panel-title">Control de Laboratorio</span>
          <span className="panel-badge">{images.length} imagenes disponibles</span>
        </div>
        <div className="lab-toolbar-body">
          <label className="lab-control">
            <span>Confianza: {conf.toFixed(2)}</span>
            <input type="range" min="0.05" max="0.95" step="0.01" value={conf} onChange={(e) => setConf(Number(e.target.value))} />
          </label>
          <label className="lab-control">
            <span>Area minima: {minArea}</span>
            <input type="range" min="500" max="30000" step="500" value={minArea} onChange={(e) => setMinArea(Number(e.target.value))} />
          </label>
          <div className="lab-actions">
            <button className="lab-btn primary" onClick={runAnalysis} disabled={loading || !images.length}>
              {loading ? "Analizando..." : "Ejecutar Analisis"}
            </button>
            <button className="lab-btn" onClick={exportReport} disabled={exporting || !images.length}>
              {exporting ? "Exportando..." : "Exportar Reporte"}
            </button>
          </div>
          {error && <div className="lab-error">{error}</div>}
          {exportError && <div className="lab-error">{exportError}</div>}
        </div>
      </section>

      <section className="lab-kpis panel">
        <div className="panel-head">
          <span className="panel-title">KPIs del Dataset</span>
          <span className="panel-badge">{analysis ? "Resultado disponible" : "Pendiente"}</span>
        </div>
        <div className="lab-kpi-strip">
          <div className="lab-kpi">
            <span>Imagenes</span>
            <strong>{analysis?.total_images ?? images.length ?? 0}</strong>
          </div>
          <div className="lab-kpi">
            <span>Detecciones</span>
            <strong>{analysis?.total_detections ?? "--"}</strong>
          </div>
          <div className="lab-kpi">
            <span>Tasa de deteccion</span>
            <strong>{analysis ? `${Math.round((analysis.kpis?.detection_rate ?? 0) * 100)}%` : "--"}</strong>
          </div>
          <div className="lab-kpi">
            <span>Confianza promedio</span>
            <strong>{analysis ? (analysis.kpis?.avg_confidence ?? 0).toFixed(2) : "--"}</strong>
          </div>
          <div className="lab-kpi">
            <span>Confianza mediana</span>
            <strong>{analysis ? (analysis.kpis?.median_confidence ?? 0).toFixed(2) : "--"}</strong>
          </div>
          <div className="lab-kpi">
            <span>Cobertura promedio</span>
            <strong>{analysis ? `${(analysis.kpis?.avg_bbox_coverage_pct ?? 0).toFixed(2)}%` : "--"}</strong>
          </div>
        </div>
      </section>

      <section className="lab-workspace">
        <div className="lab-grid panel">
          <div className="panel-head">
            <span className="panel-title">Dataset Close-up</span>
            <span className="panel-badge">Click para inspeccionar</span>
          </div>
          {loading ? (
            <div className="lab-loading">
              <div className="lab-spinner" />
              <p>Procesando inferencia close-up en el backend...</p>
            </div>
          ) : (
            <div className="lab-cards">
              {images.map((filename) => {
                const item = imageMap.get(filename);
                const avgConfidence = item?.avg_confidence ?? 0;
                const tone = toneForConfidence(avgConfidence);
                return (
                  <button
                    key={filename}
                    className={`lab-card ${selected === filename ? "is-selected" : ""}`}
                    onClick={() => setSelected(filename)}
                    type="button"
                  >
                    <div className="lab-thumb-wrap">
                      <img className="lab-thumb" src={imageSrcFor(filename, analysis, conf, minArea)} alt={filename} />
                    </div>
                    <div className="lab-card-meta">
                      <div className="lab-card-top">
                        <span className={`lab-chip ${tone}`}>{item ? `${Math.round(avgConfidence * 100)}%` : "Sin analisis"}</span>
                        <span className="lab-chip neutral">{item?.detection_count ?? 0} det</span>
                        <span className="lab-chip neutral">{lightingBadge(item?.lighting_class)}</span>
                      </div>
                      <strong>{filename}</strong>
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <aside className="lab-detail panel">
          <div className="panel-head">
            <span className="panel-title">Panel de Detalle</span>
            <span className="panel-badge">{selected ?? "Sin seleccion"}</span>
          </div>
          {selected ? (
            <div className="lab-detail-body">
              <div className="lab-detail-preview">
                <img src={selectedPreview} alt={selected} />
              </div>

              <div className="lab-detail-state">
                <span className="lab-chip neutral">{selectedImage?.lighting_class ?? "--"}</span>
                <span className="lab-chip neutral">{predominantAngle}</span>
                <span className="lab-chip neutral">{selectedImage?.detection_count ?? 0} objetos</span>
              </div>

              <div className="lab-summary">
                <div>
                  <span>Brillo medio</span>
                  <strong>{selectedImage?.mean_brightness?.toFixed(1) ?? "--"}</strong>
                </div>
                <div>
                  <span>Resolucion</span>
                  <strong>
                    {selectedImage ? `${selectedImage.image_w} x ${selectedImage.image_h}` : "--"}
                  </strong>
                </div>
                <div>
                  <span>Cobertura bbox</span>
                  <strong>{selectedImage ? `${selectedImage.bbox_coverage_pct.toFixed(2)}%` : "--"}</strong>
                </div>
                <div>
                  <span>Confianza media</span>
                  <strong>{selectedImage ? selectedImage.avg_confidence.toFixed(2) : "--"}</strong>
                </div>
              </div>

              <div className="lab-detections">
                <div className="lab-table-head">
                  <span>#</span>
                  <span>Conf</span>
                  <span>Angulo</span>
                  <span>Area</span>
                  <span>Luz</span>
                </div>
                {(selectedImage?.detections ?? []).length ? (
                  selectedImage.detections.map((det) => (
                    <div className="lab-table-row" key={`${selected}-${det.detection_idx}`}>
                      <span>{det.detection_idx + 1}</span>
                      <span>{det.confidence.toFixed(2)}</span>
                      <span>{det.angle_class}</span>
                      <span>{Math.round(det.bbox_area_px)}</span>
                      <span>{selectedImage.lighting_class}</span>
                    </div>
                  ))
                ) : (
                  <div className="lab-empty-detail">No hay detecciones para esta imagen con los umbrales actuales.</div>
                )}
              </div>
            </div>
          ) : (
            <div className="lab-empty-detail">Selecciona una imagen para revisar su detalle tecnico.</div>
          )}
        </aside>
      </section>
    </div>
  );
}
