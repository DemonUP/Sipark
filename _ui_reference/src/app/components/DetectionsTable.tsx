import { Radar } from "lucide-react";

interface Detection {
  id: string;
  label: string;
  coords: string;
  space: string;
}

interface DetectionsTableProps {
  detections: Detection[];
}

export function DetectionsTable({ detections }: DetectionsTableProps) {
  return (
    <div
      className="bg-white rounded-lg border border-[#E5E7EB] overflow-hidden flex flex-col"
      style={{
        fontFamily: 'Inter, sans-serif',
        boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
      }}
    >
      {/* Header */}
      <div className="px-5 py-4 border-b border-[#E5E7EB] flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-[#E3F2FD] flex items-center justify-center">
          <Radar className="w-4 h-4 text-[#0F5E9C]" />
        </div>
        <div>
          <h3 className="text-[#1F2937]" style={{ fontSize: '15px', fontWeight: 600 }}>
            Detecciones recientes
          </h3>
          <p className="text-[#6B7280]" style={{ fontSize: '12px' }}>
            Últimas 6 detecciones del frame actual
          </p>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto flex-1">
        <table className="w-full">
          <thead>
            <tr className="bg-[#F4F6F8]">
              <th className="text-left px-5 py-2.5 text-[#6B7280] border-b border-[#E5E7EB]" style={{ fontSize: '11px', fontWeight: 600, letterSpacing: '0.06em' }}>
                ID
              </th>
              <th className="text-left px-5 py-2.5 text-[#6B7280] border-b border-[#E5E7EB]" style={{ fontSize: '11px', fontWeight: 600, letterSpacing: '0.06em' }}>
                VEHÍCULO
              </th>
              <th className="text-left px-5 py-2.5 text-[#6B7280] border-b border-[#E5E7EB]" style={{ fontSize: '11px', fontWeight: 600, letterSpacing: '0.06em' }}>
                COORDENADAS
              </th>
              <th className="text-left px-5 py-2.5 text-[#6B7280] border-b border-[#E5E7EB]" style={{ fontSize: '11px', fontWeight: 600, letterSpacing: '0.06em' }}>
                ESPACIO
              </th>
            </tr>
          </thead>
          <tbody>
            {detections.map((d, i) => (
              <tr
                key={d.id}
                className={`border-b border-[#F4F6F8] ${i % 2 === 0 ? 'bg-white' : 'bg-[#FAFBFC]'} hover:bg-[#E3F2FD]/30 transition-colors`}
              >
                <td className="px-5 py-2.5 text-[#9CA3AF]" style={{ fontSize: '13px', fontWeight: 500 }}>
                  {d.id}
                </td>
                <td className="px-5 py-2.5 text-[#1F2937]" style={{ fontSize: '13px', fontWeight: 600 }}>
                  {d.label}
                </td>
                <td className="px-5 py-2.5">
                  <code className="bg-[#F4F6F8] text-[#4B5563] px-2 py-0.5 rounded border border-[#E5E7EB]" style={{ fontSize: '12px' }}>
                    {d.coords}
                  </code>
                </td>
                <td className="px-5 py-2.5">
                  <span
                    className="inline-flex items-center justify-center rounded px-2.5 py-0.5"
                    style={{
                      fontSize: '12px',
                      fontWeight: 700,
                      backgroundColor: '#0F5E9C',
                      color: 'white',
                      minWidth: '36px',
                      letterSpacing: '0.02em',
                    }}
                  >
                    {d.space}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
