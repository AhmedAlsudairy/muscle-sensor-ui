"use client";

import { TableIcon } from "lucide-react";

interface Reading {
  id: number;
  signal_value: number;
  signal_percentage: number;
  status: string;
  peak_value: number;
  average_value: number;
  created_at: string;
}

interface ReadingsTableProps {
  readings: Reading[];
}

export function ReadingsTable({ readings }: ReadingsTableProps) {
  // Parse numeric values safely from database strings
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "relaxed":
        return "text-emerald-400 bg-emerald-500/20";
      case "moderate":
        return "text-amber-400 bg-amber-500/20";
      case "contracted":
        return "text-red-400 bg-red-500/20";
      default:
        return "text-muted-foreground bg-secondary";
    }
  };

  if (readings.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
        <TableIcon className="w-12 h-12 mb-3 opacity-50" />
        <p className="text-sm">No readings recorded yet</p>
        <p className="text-xs mt-1">Click Connect to start recording</p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border">
            <th className="text-left py-3 px-2 text-muted-foreground font-medium">ID</th>
            <th className="text-left py-3 px-2 text-muted-foreground font-medium">Time</th>
            <th className="text-right py-3 px-2 text-muted-foreground font-medium">Raw Value</th>
            <th className="text-right py-3 px-2 text-muted-foreground font-medium">Signal %</th>
            <th className="text-center py-3 px-2 text-muted-foreground font-medium">Status</th>
            <th className="text-right py-3 px-2 text-muted-foreground font-medium">Peak</th>
            <th className="text-right py-3 px-2 text-muted-foreground font-medium">Avg</th>
          </tr>
        </thead>
        <tbody>
          {readings.map((reading) => (
            <tr
              key={reading.id}
              className="border-b border-border/50 hover:bg-secondary/50 transition-colors"
            >
              <td className="py-2 px-2 text-muted-foreground font-mono text-xs">
                #{reading.id}
              </td>
              <td className="py-2 px-2 text-foreground font-mono text-xs">
                {formatDate(reading.created_at)}
              </td>
              <td className="py-2 px-2 text-right text-foreground font-mono">
                {parseFloat(String(reading.signal_value)).toFixed(2)}
              </td>
              <td className="py-2 px-2 text-right text-primary font-semibold">
                {parseFloat(String(reading.signal_percentage)).toFixed(1)}%
              </td>
              <td className="py-2 px-2 text-center">
                <span
                  className={`inline-block px-2 py-0.5 rounded text-xs font-medium capitalize ${getStatusColor(
                    reading.status
                  )}`}
                >
                  {reading.status}
                </span>
              </td>
              <td className="py-2 px-2 text-right text-red-400 font-mono text-xs">
                {parseFloat(String(reading.peak_value)).toFixed(1)}
              </td>
              <td className="py-2 px-2 text-right text-blue-400 font-mono text-xs">
                {parseFloat(String(reading.average_value)).toFixed(1)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
