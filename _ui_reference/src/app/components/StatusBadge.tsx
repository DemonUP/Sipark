interface StatusBadgeProps {
  label: string;
  variant: "critical" | "warning" | "success" | "neutral" | "info";
}

const variantStyles: Record<StatusBadgeProps["variant"], string> = {
  critical: "bg-[#C62828] text-white",
  warning: "bg-[#FFF8E1] text-[#E65100] border border-[#FFE082]",
  success: "bg-[#E8F5E9] text-[#2E7D32] border border-[#C8E6C9]",
  neutral: "bg-[#F4F6F8] text-[#6B7280] border border-[#E5E7EB]",
  info: "bg-[#E3F2FD] text-[#0F5E9C] border border-[#BBDEFB]",
};

export function StatusBadge({ label, variant }: StatusBadgeProps) {
  return (
    <span
      className={`inline-flex items-center px-3 py-1 rounded ${variantStyles[variant]}`}
      style={{ fontSize: '11px', fontWeight: 600, fontFamily: 'Inter, sans-serif', letterSpacing: '0.04em', textTransform: 'uppercase' }}
    >
      {variant === "critical" && (
        <span className="w-[6px] h-[6px] bg-white rounded-full mr-2 animate-pulse" />
      )}
      {variant === "neutral" && (
        <span className="w-[6px] h-[6px] bg-[#9CA3AF] rounded-full mr-2" />
      )}
      {label}
    </span>
  );
}
