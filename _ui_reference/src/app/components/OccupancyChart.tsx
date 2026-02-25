import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { TrendingUp } from "lucide-react";

const data = [
  { time: "23:00", libres: 3, ocupados: 5 },
  { time: "23:05", libres: 2, ocupados: 6 },
  { time: "23:10", libres: 2, ocupados: 6 },
  { time: "23:15", libres: 1, ocupados: 7 },
  { time: "23:20", libres: 2, ocupados: 6 },
  { time: "23:25", libres: 1, ocupados: 7 },
  { time: "23:29", libres: 1, ocupados: 7 },
];

export function OccupancyChart() {
  return (
    <div
      className="bg-white rounded-lg border border-[#E5E7EB] p-6"
      style={{
        fontFamily: 'Inter, sans-serif',
        boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
      }}
    >
      <div className="flex items-center gap-3 mb-5">
        <div className="w-8 h-8 rounded-lg bg-[#E3F2FD] flex items-center justify-center">
          <TrendingUp className="w-4 h-4 text-[#0F5E9C]" />
        </div>
        <div>
          <h3 className="text-[#1F2937]" style={{ fontSize: '15px', fontWeight: 600 }}>
            Tendencia de ocupación reciente
          </h3>
          <p className="text-[#6B7280]" style={{ fontSize: '12px' }}>
            Historial de espacios libres vs. ocupados — últimas lecturas
          </p>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 8, right: 24, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" vertical={false} />
          <XAxis
            dataKey="time"
            tick={{ fontSize: 12, fill: '#6B7280', fontFamily: 'Inter, sans-serif' }}
            axisLine={{ stroke: '#E5E7EB' }}
            tickLine={false}
            dy={8}
          />
          <YAxis
            tick={{ fontSize: 12, fill: '#6B7280', fontFamily: 'Inter, sans-serif' }}
            axisLine={false}
            tickLine={false}
            domain={[0, 8]}
            dx={-4}
          />
          <Tooltip
            contentStyle={{
              fontFamily: 'Inter, sans-serif',
              fontSize: '13px',
              borderRadius: '6px',
              border: '1px solid #E5E7EB',
              boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
              padding: '8px 12px',
            }}
            labelStyle={{ fontWeight: 600, color: '#1F2937', marginBottom: 4 }}
          />
          <Legend
            wrapperStyle={{ fontSize: '12px', fontFamily: 'Inter, sans-serif', paddingTop: '16px' }}
            iconType="circle"
            iconSize={8}
          />
          <Line
            type="monotone"
            dataKey="ocupados"
            name="Ocupados"
            stroke="#0F5E9C"
            strokeWidth={2.5}
            dot={{ r: 3.5, fill: '#0F5E9C', strokeWidth: 0 }}
            activeDot={{ r: 5, fill: '#0F5E9C', stroke: '#E3F2FD', strokeWidth: 3 }}
          />
          <Line
            type="monotone"
            dataKey="libres"
            name="Libres"
            stroke="#2E7D32"
            strokeWidth={2.5}
            dot={{ r: 3.5, fill: '#2E7D32', strokeWidth: 0 }}
            activeDot={{ r: 5, fill: '#2E7D32', stroke: '#E8F5E9', strokeWidth: 3 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
