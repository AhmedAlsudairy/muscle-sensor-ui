"use client";

import { useEffect, useState } from "react";
import { Activity, Zap, Heart } from "lucide-react";

interface MuscleIndicatorProps {
  signalStrength: number;
}

export function MuscleIndicator({ signalStrength }: MuscleIndicatorProps) {
  const [status, setStatus] = useState<"relaxed" | "moderate" | "contracted">(
    "relaxed"
  );

  useEffect(() => {
    if (signalStrength < 30) {
      setStatus("relaxed");
    } else if (signalStrength < 70) {
      setStatus("moderate");
    } else {
      setStatus("contracted");
    }
  }, [signalStrength]);

  const statusConfig = {
    relaxed: {
      color: "bg-emerald-500",
      textColor: "text-emerald-400",
      label: "Relaxed",
      icon: Heart,
    },
    moderate: {
      color: "bg-amber-500",
      textColor: "text-amber-400",
      label: "Moderate Activity",
      icon: Activity,
    },
    contracted: {
      color: "bg-red-500",
      textColor: "text-red-400",
      label: "Contracted",
      icon: Zap,
    },
  };

  const config = statusConfig[status];
  const Icon = config.icon;

  return (
    <div className="flex flex-col items-center gap-4">
      {/* Muscle visualization */}
      <div className="relative">
        <div
          className={`w-32 h-32 rounded-full border-4 ${config.color} border-opacity-30 flex items-center justify-center transition-all duration-300`}
          style={{
            boxShadow: `0 0 ${signalStrength / 2}px ${signalStrength / 4}px ${
              status === "contracted"
                ? "rgba(239, 68, 68, 0.3)"
                : status === "moderate"
                ? "rgba(251, 191, 36, 0.3)"
                : "rgba(52, 211, 153, 0.3)"
            }`,
          }}
        >
          <div
            className={`w-24 h-24 rounded-full ${config.color} opacity-60 flex items-center justify-center transition-all duration-300`}
            style={{
              transform: `scale(${0.5 + signalStrength / 200})`,
            }}
          >
            <Icon className={`w-10 h-10 text-white`} />
          </div>
        </div>

        {/* Pulse animation */}
        {status !== "relaxed" && (
          <div
            className={`absolute inset-0 w-32 h-32 rounded-full ${config.color} opacity-20 animate-ping`}
          />
        )}
      </div>

      {/* Status label */}
      <div className={`text-lg font-semibold ${config.textColor}`}>
        {config.label}
      </div>

      {/* Strength bar */}
      <div className="w-full max-w-xs">
        <div className="flex justify-between text-xs text-muted-foreground mb-1">
          <span>0%</span>
          <span>Signal Strength</span>
          <span>100%</span>
        </div>
        <div className="h-3 bg-secondary rounded-full overflow-hidden">
          <div
            className={`h-full ${config.color} transition-all duration-150 rounded-full`}
            style={{ width: `${signalStrength}%` }}
          />
        </div>
      </div>
    </div>
  );
}
