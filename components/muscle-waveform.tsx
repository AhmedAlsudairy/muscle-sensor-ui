"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  ReferenceLine,
  Tooltip,
  CartesianGrid,
  Label,
} from "recharts";

interface DataPoint {
  time: number;
  value: number;
}

interface MuscleWaveformProps {
  data: DataPoint[];
}

interface TooltipPayloadItem {
  value: number;
}

function CustomTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: TooltipPayloadItem[];
  label?: number;
}) {
  if (!active || !payload || payload.length === 0) return null;
  return (
    <div className="rounded-lg border border-border bg-card px-3 py-2 shadow-lg text-xs">
      <p className="text-muted-foreground mb-1">Sample&nbsp;<span className="font-medium text-foreground">{label}</span></p>
      <p className="text-emerald-400 font-semibold">{payload[0].value.toFixed(1)}%</p>
    </div>
  );
}

export function MuscleWaveform({ data }: MuscleWaveformProps) {
  // Pad to 100 points so the chart width stays stable
  const padded: DataPoint[] =
    data.length >= 100
      ? data.slice(-100)
      : [
          ...Array.from({ length: 100 - data.length }, (_, i) => ({
            time: i,
            value: 0,
          })),
          ...data,
        ];

  return (
    <div className="h-full w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={padded} margin={{ top: 8, right: 16, bottom: 28, left: 44 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
          <XAxis
            dataKey="time"
            tick={{ fill: "#6b7280", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "#374151" }}
            interval={19}
          >
            <Label
              value="Sample Index"
              offset={-12}
              position="insideBottom"
              style={{ fill: "#9ca3af", fontSize: 11 }}
            />
          </XAxis>
          <YAxis
            domain={[0, 100]}
            tick={{ fill: "#6b7280", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "#374151" }}
            ticks={[0, 25, 50, 75, 100]}
            tickFormatter={(v) => `${v}%`}
            width={40}
          >
            <Label
              value="Signal (%)"
              angle={-90}
              position="insideLeft"
              offset={10}
              style={{ fill: "#9ca3af", fontSize: 11, textAnchor: "middle" }}
            />
          </YAxis>
          <ReferenceLine y={50} stroke="#374151" strokeDasharray="4 4" label={{ value: "50%", fill: "#4b5563", fontSize: 10, position: "right" }} />
          <Tooltip content={<CustomTooltip />} cursor={{ stroke: "#4b5563", strokeWidth: 1 }} />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#34d399"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
            activeDot={{ r: 4, fill: "#34d399", stroke: "#065f46", strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
