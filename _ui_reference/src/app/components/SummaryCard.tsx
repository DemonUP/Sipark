interface SummaryCardProps {
  title: string;
  value: number | string;
  color?: string;
  subtitle?: string;
  showBar?: boolean;
  barPercent?: number;
  barColor?: string;
  icon?: React.ReactNode;
}

export function SummaryCard({
  title,
  value,
  color = "#1F2937",
  subtitle,
  showBar,
  barPercent,
  barColor = "#0F5E9C",
  icon,
}: SummaryCardProps) {
  return (
    <div
      className="bg-white rounded-lg border border-[#E5E7EB] p-5 flex flex-col gap-3"
      style={{
        fontFamily: 'Inter, sans-serif',
        boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
      }}
    >
      <div className="flex items-center justify-between">
        <span className="text-[#6B7280]" style={{ fontSize: '13px', fontWeight: 500, letterSpacing: '0.01em' }}>
          {title}
        </span>
        {icon && <span className="text-[#0F5E9C]/40">{icon}</span>}
      </div>
      <span style={{ fontSize: '36px', fontWeight: 700, color, lineHeight: 1 }}>
        {value}
      </span>
      {showBar && barPercent !== undefined && (
        <div className="mt-auto">
          <div className="w-full h-[6px] bg-[#F4F6F8] rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{ width: `${barPercent}%`, backgroundColor: barColor }}
            />
          </div>
          <span className="text-[#9CA3AF] mt-1.5 block" style={{ fontSize: '11px' }}>
            {barPercent}% de capacidad utilizada
          </span>
        </div>
      )}
      {subtitle && (
        <span className="text-[#9CA3AF] mt-auto" style={{ fontSize: '11px' }}>
          {subtitle}
        </span>
      )}
    </div>
  );
}
