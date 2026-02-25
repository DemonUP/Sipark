import { Bike, CircleCheck } from "lucide-react";

interface SpaceCardProps {
  id: string;
  status: "OCUPADO" | "LIBRE";
  count: number;
}

export function SpaceCard({ id, status }: SpaceCardProps) {
  const isOccupied = status === "OCUPADO";

  return (
    <div
      className="bg-white rounded-lg border border-[#E5E7EB] overflow-hidden transition-all hover:shadow-md"
      style={{
        fontFamily: 'Inter, sans-serif',
        boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
      }}
    >
      {/* Top color stripe */}
      <div
        className="h-[4px] w-full"
        style={{ backgroundColor: isOccupied ? '#C62828' : '#2E7D32' }}
      />
      <div className="px-3 py-3 flex flex-col items-center gap-2">
        <span className="text-[#1F2937]" style={{ fontSize: '15px', fontWeight: 700 }}>
          {id}
        </span>
        <div className={`w-8 h-8 rounded-full flex items-center justify-center ${isOccupied ? 'bg-[#FFEBEE]' : 'bg-[#E8F5E9]'}`}>
          {isOccupied ? (
            <Bike className="w-4 h-4 text-[#C62828]" />
          ) : (
            <CircleCheck className="w-4 h-4 text-[#2E7D32]" />
          )}
        </div>
        <span
          className="w-full text-center py-1 rounded"
          style={{
            fontSize: '10px',
            fontWeight: 700,
            letterSpacing: '0.06em',
            color: isOccupied ? '#C62828' : '#2E7D32',
            backgroundColor: isOccupied ? '#FFEBEE' : '#E8F5E9',
          }}
        >
          {status}
        </span>
      </div>
    </div>
  );
}
