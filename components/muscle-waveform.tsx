"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

interface DataPoint {
  time: number;
  value: number;
}

interface MuscleWaveformProps {
  data: DataPoint[];
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
        <LineChart data={padded}>
          <XAxis dataKey="time" hide />
          <YAxis domain={[0, 100]} hide />
          <ReferenceLine y={50} stroke="#374151" strokeDasharray="3 3" />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#34d399"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
