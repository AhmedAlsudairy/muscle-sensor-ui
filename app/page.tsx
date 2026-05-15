"use client";

import { useEffect, useState, useRef } from "react";
import {
  Activity,
  Zap,
  Timer,
  TrendingUp,
  Cpu,
  Radio,
  Power,
  CircleDot,
} from "lucide-react";
import { MuscleWaveform } from "@/components/muscle-waveform";
import { MuscleIndicator } from "@/components/muscle-indicator";
import { StatsCard } from "@/components/stats-card";
import { HistoryChart } from "@/components/history-chart";

interface HistoryDataPoint {
  time: string;
  value: number;
}

export default function MuscleSensorDashboard() {
  const [signalStrength, setSignalStrength] = useState(25);
  const [isConnected, setIsConnected] = useState(true);
  const [historyData, setHistoryData] = useState<HistoryDataPoint[]>([]);
  const [peakValue, setPeakValue] = useState(0);
  const [avgValue, setAvgValue] = useState(0);
  const [sessionTime, setSessionTime] = useState(0);
  const valuesRef = useRef<number[]>([]);

  useEffect(() => {
    // Simulate signal strength changes
    const signalInterval = setInterval(() => {
      if (isConnected) {
        const newValue = Math.min(
          100,
          Math.max(
            0,
            signalStrength + Math.random() * 30 - 15 + Math.sin(Date.now() / 1000) * 10
          )
        );
        setSignalStrength(newValue);

        // Track values for statistics
        valuesRef.current.push(newValue);
        if (valuesRef.current.length > 100) {
          valuesRef.current = valuesRef.current.slice(-100);
        }

        // Update peak
        if (newValue > peakValue) {
          setPeakValue(newValue);
        }

        // Update average
        const avg =
          valuesRef.current.reduce((a, b) => a + b, 0) / valuesRef.current.length;
        setAvgValue(avg);
      }
    }, 100);

    // Update history data every second
    const historyInterval = setInterval(() => {
      if (isConnected) {
        setHistoryData((prev) => {
          const now = new Date();
          const timeStr = `${now.getMinutes()}:${now.getSeconds().toString().padStart(2, "0")}`;
          const newData = [
            ...prev.slice(-29),
            { time: timeStr, value: signalStrength },
          ];
          return newData;
        });
      }
    }, 1000);

    // Session timer
    const timerInterval = setInterval(() => {
      if (isConnected) {
        setSessionTime((prev) => prev + 1);
      }
    }, 1000);

    return () => {
      clearInterval(signalInterval);
      clearInterval(historyInterval);
      clearInterval(timerInterval);
    };
  }, [isConnected, signalStrength, peakValue]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="min-h-screen bg-background p-4 md:p-6">
      {/* Header */}
      <header className="mb-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold text-foreground flex items-center gap-3">
              <Activity className="w-8 h-8 text-primary" />
              Muscle Sensor Monitor
            </h1>
            <p className="text-muted-foreground mt-1">
              Real-time EMG signal monitoring dashboard
            </p>
          </div>
          <div className="flex items-center gap-4">
            {/* Connection Status */}
            <div
              className={`flex items-center gap-2 px-4 py-2 rounded-lg border ${
                isConnected
                  ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-400"
                  : "border-red-500/50 bg-red-500/10 text-red-400"
              }`}
            >
              <CircleDot
                className={`w-4 h-4 ${isConnected ? "animate-pulse" : ""}`}
              />
              <span className="text-sm font-medium">
                {isConnected ? "Connected" : "Disconnected"}
              </span>
            </div>
            {/* Connect/Disconnect Button */}
            <button
              onClick={() => setIsConnected(!isConnected)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                isConnected
                  ? "bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/30"
                  : "bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 border border-emerald-500/30"
              }`}
            >
              <Power className="w-4 h-4" />
              {isConnected ? "Disconnect" : "Connect"}
            </button>
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Waveform & Stats */}
        <div className="lg:col-span-2 space-y-6">
          {/* Real-time Waveform */}
          <div className="bg-card rounded-xl border border-border p-4 md:p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Radio className="w-5 h-5 text-primary" />
                <h2 className="text-lg font-semibold text-foreground">
                  EMG Signal Waveform
                </h2>
              </div>
              <span className="text-xs text-muted-foreground bg-secondary px-2 py-1 rounded">
                Live
              </span>
            </div>
            <div className="h-48 md:h-64">
              <MuscleWaveform />
            </div>
          </div>

          {/* Stats Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatsCard
              title="Current"
              value={signalStrength.toFixed(0)}
              unit="%"
              icon={Zap}
              color="emerald"
            />
            <StatsCard
              title="Peak"
              value={peakValue.toFixed(0)}
              unit="%"
              icon={TrendingUp}
              color="red"
            />
            <StatsCard
              title="Average"
              value={avgValue.toFixed(0)}
              unit="%"
              icon={Activity}
              color="blue"
            />
            <StatsCard
              title="Session"
              value={formatTime(sessionTime)}
              icon={Timer}
              color="amber"
            />
          </div>

          {/* History Chart */}
          <div className="bg-card rounded-xl border border-border p-4 md:p-6">
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp className="w-5 h-5 text-primary" />
              <h2 className="text-lg font-semibold text-foreground">
                Signal History
              </h2>
            </div>
            <div className="h-48 md:h-56">
              <HistoryChart data={historyData} />
            </div>
          </div>
        </div>

        {/* Right Column - Muscle Indicator & Device Info */}
        <div className="space-y-6">
          {/* Muscle Status */}
          <div className="bg-card rounded-xl border border-border p-6">
            <div className="flex items-center gap-2 mb-6">
              <Activity className="w-5 h-5 text-primary" />
              <h2 className="text-lg font-semibold text-foreground">
                Muscle Status
              </h2>
            </div>
            <MuscleIndicator signalStrength={signalStrength} />
          </div>

          {/* Device Information */}
          <div className="bg-card rounded-xl border border-border p-6">
            <div className="flex items-center gap-2 mb-4">
              <Cpu className="w-5 h-5 text-primary" />
              <h2 className="text-lg font-semibold text-foreground">
                Device Info
              </h2>
            </div>
            <div className="space-y-3">
              <div className="flex justify-between py-2 border-b border-border">
                <span className="text-muted-foreground">Sensor Type</span>
                <span className="text-foreground font-medium">EMG v3</span>
              </div>
              <div className="flex justify-between py-2 border-b border-border">
                <span className="text-muted-foreground">Controller</span>
                <span className="text-foreground font-medium">Arduino UNO</span>
              </div>
              <div className="flex justify-between py-2 border-b border-border">
                <span className="text-muted-foreground">Sample Rate</span>
                <span className="text-foreground font-medium">500 Hz</span>
              </div>
              <div className="flex justify-between py-2 border-b border-border">
                <span className="text-muted-foreground">Resolution</span>
                <span className="text-foreground font-medium">10-bit ADC</span>
              </div>
              <div className="flex justify-between py-2">
                <span className="text-muted-foreground">Power</span>
                <span className="text-foreground font-medium">12V DC</span>
              </div>
            </div>
          </div>

          {/* Threshold Settings */}
          <div className="bg-card rounded-xl border border-border p-6">
            <h2 className="text-lg font-semibold text-foreground mb-4">
              Threshold Levels
            </h2>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-muted-foreground">Relaxed</span>
                  <span className="text-emerald-400">0-30%</span>
                </div>
                <div className="h-2 bg-emerald-500/30 rounded-full" />
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-muted-foreground">Moderate</span>
                  <span className="text-amber-400">30-70%</span>
                </div>
                <div className="h-2 bg-amber-500/30 rounded-full" />
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-muted-foreground">Contracted</span>
                  <span className="text-red-400">70-100%</span>
                </div>
                <div className="h-2 bg-red-500/30 rounded-full" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
