"use client";

import { useEffect, useState, useRef, useCallback } from "react";
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
  Download,
} from "lucide-react";
import { MuscleWaveform } from "@/components/muscle-waveform";
import { MuscleIndicator } from "@/components/muscle-indicator";
import { StatsCard } from "@/components/stats-card";
import { HistoryChart } from "@/components/history-chart";

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
  const [signalStrength, setSignalStrength] = useState(25);
  const [isConnected, setIsConnected] = useState(false);
  const [historyData, setHistoryData] = useState<HistoryDataPoint[]>([]);
  const [peakValue, setPeakValue] = useState(0);
  const [avgValue, setAvgValue] = useState(0);
  const [sessionTime, setSessionTime] = useState(0);
  const [dbStatus, setDbStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [totalReadings, setTotalReadings] = useState(0);
  const [isSaving, setIsSaving] = useState(false);
  const valuesRef = useRef<number[]>([]);

  // Setup database on mount
  useEffect(() => {
    const setupDb = async () => {
      setDbStatus("loading");
      try {
        const response = await fetch("/api/setup", { method: "POST" });
        if (response.ok) {
          setDbStatus("ready");
          // Load existing readings
          loadReadings();
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

  // Load readings from database
  const loadReadings = async () => {
    try {
      const response = await fetch("/api/readings?limit=100");
      const data = await response.json();
      if (data.success && data.readings) {
        // Convert readings to history data
        const history: HistoryDataPoint[] = data.readings.map((r: Reading) => {
          const date = new Date(r.created_at);
          return {
            time: `${date.getMinutes()}:${date.getSeconds().toString().padStart(2, "0")}`,
            value: parseFloat(String(r.signal_percentage)),
          };
        });
        setHistoryData(history.slice(-30));
        setTotalReadings(parseInt(data.stats.total_readings) || 0);
        
        if (data.stats.max_signal) {
          setPeakValue(parseFloat(data.stats.max_signal));
        }
        if (data.stats.avg_signal) {
          setAvgValue(parseFloat(data.stats.avg_signal));
        }
      }
    } catch (error) {
      console.error("Failed to load readings:", error);
    }
  };

  // Get muscle status based on signal strength
  const getMuscleStatus = useCallback((value: number): string => {
    if (value < 30) return "relaxed";
    if (value < 70) return "moderate";
    return "contracted";
  }, []);

  // Save reading to database
  const saveReading = useCallback(async (value: number, peak: number, avg: number) => {
    if (isSaving) return;
    setIsSaving(true);
    try {
      await fetch("/api/readings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          signal_value: value,
          signal_percentage: value,
          status: getMuscleStatus(value),
          peak_value: peak,
          average_value: avg,
        }),
      });
      setTotalReadings((prev) => prev + 1);
    } catch (error) {
      console.error("Failed to save reading:", error);
    } finally {
      setIsSaving(false);
    }
  }, [isSaving, getMuscleStatus]);

  // Clear all readings
  const clearReadings = async () => {
    if (!confirm("Are you sure you want to clear all readings?")) return;
    try {
      await fetch("/api/readings", { method: "DELETE" });
      setHistoryData([]);
      setPeakValue(0);
      setAvgValue(0);
      setTotalReadings(0);
      valuesRef.current = [];
    } catch (error) {
      console.error("Failed to clear readings:", error);
    }
  };

  useEffect(() => {
    if (!isConnected || dbStatus !== "ready") return;

    // Simulate signal strength changes
    const signalInterval = setInterval(() => {
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
    }, 100);

    // Update history data and save to database every second
    const historyInterval = setInterval(() => {
      setHistoryData((prev) => {
        const now = new Date();
        const timeStr = `${now.getMinutes()}:${now.getSeconds().toString().padStart(2, "0")}`;
        const newData = [
          ...prev.slice(-29),
          { time: timeStr, value: signalStrength },
        ];
        return newData;
      });
      
      // Save to database every second
      saveReading(signalStrength, peakValue, avgValue);
    }, 1000);

    // Session timer
    const timerInterval = setInterval(() => {
      setSessionTime((prev) => prev + 1);
    }, 1000);

    return () => {
      clearInterval(signalInterval);
      clearInterval(historyInterval);
      clearInterval(timerInterval);
    };
  }, [isConnected, dbStatus, signalStrength, peakValue, avgValue, saveReading]);

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
              <div className="flex justify-between py-2 border-b border-border">
                <span className="text-muted-foreground">Power</span>
                <span className="text-foreground font-medium">12V DC</span>
              </div>
              <div className="flex justify-between py-2">
                <span className="text-muted-foreground">Database</span>
                <span className={`font-medium ${
                  dbStatus === "ready" ? "text-emerald-400" : "text-amber-400"
                }`}>
                  {dbStatus === "ready" ? "Neon PostgreSQL" : "Connecting..."}
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
    </div>
  );
}
