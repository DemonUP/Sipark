import { AlertTriangle, Wifi, Camera } from "lucide-react";

interface CriticalBannerProps {
  visible: boolean;
  message?: string;
}

export function CriticalBanner({ visible, message }: CriticalBannerProps) {
  if (!visible) return null;

  return (
    <div
      className="w-full bg-[#C62828] px-6 py-3 flex items-center justify-center gap-3"
      style={{ fontFamily: 'Inter, sans-serif' }}
    >
      <AlertTriangle className="w-[18px] h-[18px] text-white shrink-0" />
      <p className="text-white" style={{ fontSize: '13px', fontWeight: 500, lineHeight: 1.4 }}>
        {message || (
          <>
            <span style={{ fontWeight: 700 }}>Estado crítico:</span>{" "}
            No se ha establecido conexión con el sistema. Verifique la red y el estado de la cámara.
          </>
        )}
      </p>
      <div className="hidden sm:flex items-center gap-3 ml-4">
        <span className="flex items-center gap-1.5 bg-white/15 px-2.5 py-1 rounded" style={{ fontSize: '11px', fontWeight: 600, color: 'white' }}>
          <Wifi className="w-3.5 h-3.5" /> Red
        </span>
        <span className="flex items-center gap-1.5 bg-white/15 px-2.5 py-1 rounded" style={{ fontSize: '11px', fontWeight: 600, color: 'white' }}>
          <Camera className="w-3.5 h-3.5" /> Cámara
        </span>
      </div>
    </div>
  );
}
