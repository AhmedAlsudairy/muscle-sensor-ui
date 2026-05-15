"use client";

import { useEffect, useState, useRef } from "react";
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

export function MuscleWaveform() {
  const [data, setData] = useState<DataPoint[]>([]);
  const timeRef = useRef(0);

  useEffect(() => {
    // Initialize with empty data
    const initialData: DataPoint[] = Array.from({ length: 100 }, (_, i) => ({
      time: i,
      value: 0,
    }));
    setData(initialData);

    // Simulate EMG signal
    const interval = setInterval(() => {
      timeRef.current += 1;
      setData((prevData) => {
        const newData = [...prevData.slice(1)];
        // Simulate realistic EMG signal with noise and muscle contractions
        const baseNoise = Math.random() * 20 - 10;
        const muscleSignal =
          Math.sin(timeRef.current * 0.1) * 30 +
          Math.sin(timeRef.current * 0.3) * 15 +
          Math.sin(timeRef.current * 0.7) * 10;
        const spike = Math.random() > 0.9 ? Math.random() * 40 : 0;

        newData.push({
          time: timeRef.current,
          value: 50 + baseNoise + muscleSignal + spike,
        });
        return newData;
      });
    }, 50);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-full w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
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
