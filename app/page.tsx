"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Activity,
  Zap,
  Timer,
  TrendingUp,
  Cpu,
  Radio,
  Power,
  CircleDot,
  Database,
  Trash2,
  Wifi,
} from "lucide-react";
import { MuscleWaveform } from "@/components/muscle-waveform";
import { MuscleIndicator } from "@/components/muscle-indicator";
import { StatsCard } from "@/components/stats-card";
import { HistoryChart } from "@/components/history-chart";
import { ReadingsTable } from "@/components/readings-table";

interface HistoryDataPoint {
  time: string;
  value: number;
}

interface Reading {
  id: number;
  signal_value: number;
  signal_percentage: number;
  status: string;
  peak_value: number;
  average_value: number;
  created_at: string;
}

interface DbStats {
  total_readings: string;
  max_signal: string;
  avg_signal: string;
  first_reading: string;
  last_reading: string;
}

export default function MuscleSensorDashboard() {
  const [signalStrength, setSignalStrength] = useState(0);
  const [isConnected, setIsConnected] = useState(false);
  const [historyData, setHistoryData] = useState<HistoryDataPoint[]>([]);
  const [waveformData, setWaveformData] = useState<{ time: number; value: number }[]>([]);
  const [peakValue, setPeakValue] = useState(0);
  const [avgValue, setAvgValue] = useState(0);
  const [sessionTime, setSessionTime] = useState(0);
  const [dbStatus, setDbStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [totalReadings, setTotalReadings] = useState(0);
  const [rawReadings, setRawReadings] = useState<Reading[]>([]);
  const [lastReading, setLastReading] = useState<string | null>(null);

  // Setup database on mount
  useEffect(() => {
    const setupDb = async () => {
      setDbStatus("loading");
      try {
        const response = await fetch("/api/setup", { method: "POST" });
        if (response.ok) {
          setDbStatus("ready");
          fetchReadings();
        } else {
          setDbStatus("error");
        }
      } catch (error) {
        console.error("DB setup error:", error);
        setDbStatus("error");
      }
    };
    setupDb();
  }, []);

  // Fetch latest readings from the database
  const fetchReadings = useCallback(async () => {
    try {
      const response = await fetch("/api/readings?limit=100");
      const data = await response.json();
      if (data.success && data.readings) {
        setRawReadings(data.readings);

        const history: HistoryDataPoint[] = data.readings.map((r: Reading) => {
          const date = new Date(r.created_at);
          return {
            time: `${date.getMinutes()}:${date.getSeconds().toString().padStart(2, "0")}`,
            value: parseFloat(String(r.signal_percentage)),
          };
        });
        setHistoryData(history.slice(-30));

        // Build waveform data from DB readings (indexed 0‥n)
        const wf = data.readings.map((r: Reading, i: number) => ({
          time: i,
          value: parseFloat(String(r.signal_percentage)),
        }));
        setWaveformData(wf.slice(-100));

        // Latest reading → current signal
        if (data.readings.length > 0) {
          const latest: Reading = data.readings[data.readings.length - 1];
          setSignalStrength(parseFloat(String(latest.signal_percentage)));
        }

        setTotalReadings(parseInt(data.stats.total_readings) || 0);
        if (data.stats.max_signal) setPeakValue(parseFloat(data.stats.max_signal));
        if (data.stats.avg_signal) setAvgValue(parseFloat(data.stats.avg_signal));
        if (data.stats.last_reading) setLastReading(data.stats.last_reading);
      }
    } catch (error) {
      console.error("Failed to fetch readings:", error);
    }
  }, []);

  const piStatus = (() => {
    if (!lastReading) return "offline" as const;
    const diffMs = Date.now() - new Date(lastReading).getTime();
    if (diffMs < 10_000) return "live" as const;
    if (diffMs < 60_000) return "delayed" as const;
    return "offline" as const;
  })();

  // Two-state muscle indicator: relaxed (<30%) = green, active (≥30%) = red
  const muscleState: "relaxed" | "active" = signalStrength < 30 ? "relaxed" : "active";

  // Clear all readings
  const clearReadings = async () => {
    if (!confirm("Are you sure you want to clear all readings?")) return;
    try {
      await fetch("/api/readings", { method: "DELETE" });
      setHistoryData([]);
      setWaveformData([]);
      setSignalStrength(0);
      setPeakValue(0);
      setAvgValue(0);
      setTotalReadings(0);
      setRawReadings([]);
    } catch (error) {
      console.error("Failed to clear readings:", error);
    }
  };

  // Poll DB every 2 s while connected
  useEffect(() => {
    if (!isConnected || dbStatus !== "ready") return;

    fetchReadings(); // immediate fetch on connect
    const pollInterval = setInterval(fetchReadings, 2000);
    const timerInterval = setInterval(() => setSessionTime((prev) => prev + 1), 1000);

    return () => {
      clearInterval(pollInterval);
      clearInterval(timerInterval);
    };
  }, [isConnected, dbStatus, fetchReadings]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };


  return (
    <div className="min-h-screen bg-background p-4 md:p-6">
      {/* Project Title Banner */}
      <div className="mb-4 rounded-xl border border-primary/30 bg-primary/5 px-6 py-4 text-center">
        <h1 className="text-xl md:text-2xl font-bold text-primary tracking-wide">
          AI-Based Muscle Fatigue Detection using EMG and IoT
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Presented by&nbsp;<span className="font-semibold text-foreground">Al Zahraa Rashid</span>
        </p>
      </div>

      {/* Header */}
      <header className="mb-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-2xl md:text-3xl font-bold text-foreground flex items-center gap-3">
              <Activity className="w-8 h-8 text-primary" />
              Muscle Sensor Monitor
            </h1>
            <p className="text-muted-foreground mt-1">
              EMG signal monitoring dashboard
            </p>
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            {/* Database Status */}
            <div
              className={`flex items-center gap-2 px-3 py-2 rounded-lg border ${
                dbStatus === "ready"
                  ? "border-blue-500/50 bg-blue-500/10 text-blue-400"
                  : dbStatus === "loading"
                  ? "border-amber-500/50 bg-amber-500/10 text-amber-400"
                  : dbStatus === "error"
                  ? "border-red-500/50 bg-red-500/10 text-red-400"
                  : "border-border bg-card text-muted-foreground"
              }`}
            >
              <Database className={`w-4 h-4 ${dbStatus === "loading" ? "animate-pulse" : ""}`} />
              <span className="text-sm font-medium">
                {dbStatus === "ready" ? `DB: ${totalReadings} readings` : 
                 dbStatus === "loading" ? "Connecting..." :
                 dbStatus === "error" ? "DB Error" : "DB Idle"}
              </span>
            </div>
            {/* Pi / Arduino Live Status */}
            <div
              className={`flex items-center gap-2 px-3 py-2 rounded-lg border ${
                piStatus === "live"
                  ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-400"
                  : piStatus === "delayed"
                  ? "border-amber-500/50 bg-amber-500/10 text-amber-400"
                  : "border-red-500/50 bg-red-500/10 text-red-400"
              }`}
            >
              <Wifi
                className={`w-4 h-4 ${piStatus === "live" ? "animate-pulse" : ""}`}
              />
              <span className="text-sm font-medium">
                {piStatus === "live" ? "Pi: Live" : piStatus === "delayed" ? "Pi: Delayed" : "Pi: Offline"}
              </span>
            </div>
            {/* Connection Status */}
            <div
              className={`flex items-center gap-2 px-3 py-2 rounded-lg border ${
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
            {/* Clear Data Button */}
            <button
              onClick={clearReadings}
              className="flex items-center gap-2 px-3 py-2 rounded-lg font-medium transition-colors bg-secondary text-muted-foreground hover:text-foreground border border-border"
              title="Clear all readings"
            >
              <Trash2 className="w-4 h-4" />
            </button>
            {/* Connect/Disconnect Button */}
            <button
              onClick={() => setIsConnected(!isConnected)}
              disabled={dbStatus !== "ready"}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                dbStatus !== "ready"
                  ? "bg-secondary text-muted-foreground cursor-not-allowed"
                  : isConnected
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
              <span className={`text-xs px-2 py-1 rounded ${
                isConnected 
                  ? "text-emerald-400 bg-emerald-500/20" 
                  : "text-muted-foreground bg-secondary"
              }`}>
                {isConnected ? "Live" : "Paused"}
              </span>
            </div>
            <div className="h-48 md:h-64">
              <MuscleWaveform data={waveformData} />
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
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-primary" />
                <h2 className="text-lg font-semibold text-foreground">
                  Signal History
                </h2>
              </div>
              <span className="text-xs text-muted-foreground">
                Last 30 readings
              </span>
            </div>
            <div className="h-48 md:h-56">
              <HistoryChart data={historyData} />
            </div>
          </div>
        </div>

        {/* Right Column - Muscle Indicator & Device Info */}
        <div className="space-y-6">
          {/* Muscle State — two-LED indicator */}
          <div className="bg-card rounded-xl border border-border p-6">
            <div className="flex items-center gap-2 mb-6">
              <Activity className="w-5 h-5 text-primary" />
              <h2 className="text-lg font-semibold text-foreground">
                Muscle State
              </h2>
            </div>

            {/* LED circles */}
            <div className="flex justify-center gap-10 mb-4">
              {/* Green LED */}
              <div className="flex flex-col items-center gap-3">
                <div
                  className={`w-20 h-20 rounded-full border-4 transition-all duration-300 ${
                    muscleState === "relaxed"
                      ? "bg-emerald-400 border-emerald-300 shadow-[0_0_28px_8px_rgba(52,211,153,0.55)]"
                      : "bg-emerald-950 border-emerald-900/40"
                  }`}
                />
                <span
                  className={`text-sm font-semibold tracking-wide ${
                    muscleState === "relaxed" ? "text-emerald-400" : "text-muted-foreground/40"
                  }`}
                >
                  RELAXED
                </span>
              </div>

              {/* Red LED */}
              <div className="flex flex-col items-center gap-3">
                <div
                  className={`w-20 h-20 rounded-full border-4 transition-all duration-300 ${
                    muscleState === "active"
                      ? "bg-red-400 border-red-300 shadow-[0_0_28px_8px_rgba(248,113,113,0.55)]"
                      : "bg-red-950 border-red-900/40"
                  }`}
                />
                <span
                  className={`text-sm font-semibold tracking-wide ${
                    muscleState === "active" ? "text-red-400" : "text-muted-foreground/40"
                  }`}
                >
                  ACTIVE
                </span>
              </div>
            </div>

            {/* State label */}
            <div className="text-center">
              <span
                className={`inline-block px-4 py-1.5 rounded-full text-sm font-medium border ${
                  muscleState === "relaxed"
                    ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
                    : "bg-red-500/10 border-red-500/30 text-red-400"
                }`}
              >
                {muscleState === "relaxed" ? "Muscle Relaxed" : "Muscle Active"}
                {" — "}{signalStrength.toFixed(1)}%
              </span>
            </div>
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
              <div className="flex justify-between py-2 border-b border-border">
                <span className="text-muted-foreground">Power</span>
                <span className="text-foreground font-medium">12V DC</span>
              </div>
              <div className="flex justify-between py-2 border-b border-border">
                <span className="text-muted-foreground">Database</span>
                <span className={`font-medium ${
                  dbStatus === "ready" ? "text-emerald-400" : "text-amber-400"
                }`}>
                  {dbStatus === "ready" ? "Neon PostgreSQL" : "Connecting..."}
                </span>
              </div>
              <div className="flex justify-between py-2">
                <span className="text-muted-foreground">Pi API</span>
                <span className={`font-medium flex items-center gap-1.5 ${
                  piStatus === "live" ? "text-emerald-400" : piStatus === "delayed" ? "text-amber-400" : "text-red-400"
                }`}>
                  <span className={`inline-block w-2 h-2 rounded-full ${
                    piStatus === "live" ? "bg-emerald-400 animate-pulse" : piStatus === "delayed" ? "bg-amber-400" : "bg-red-400"
                  }`} />
                  {piStatus === "live" ? "Streaming" : piStatus === "delayed" ? "Delayed" : "Offline"}
                </span>
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

      {/* Raw Database Readings Table */}
      <div className="mt-6 bg-card rounded-xl border border-border p-4 md:p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Database className="w-5 h-5 text-primary" />
            <h2 className="text-lg font-semibold text-foreground">
              Raw Database Readings
            </h2>
          </div>
          <span className="text-xs text-muted-foreground">
            Last {rawReadings.length} records from database
          </span>
        </div>
        <div className="max-h-96 overflow-y-auto">
          <ReadingsTable readings={[...rawReadings].reverse()} />
        </div>
      </div>
    </div>
  );
}
