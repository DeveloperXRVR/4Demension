"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  Cpu,
  Eye,
  ArrowRight,
  RotateCcw,
} from "lucide-react";

interface Job {
  id: string;
  status: "uploading" | "queued" | "processing" | "completed" | "failed";
  videoFilename: string;
  quality: string;
  progress: number;
  message: string;
  modelUrl?: string;
  createdAt: number;
}

const statusConfig = {
  uploading: {
    icon: <Loader2 className="w-6 h-6 animate-spin text-blue-400" />,
    color: "text-blue-400",
    bg: "bg-blue-500/10",
    border: "border-blue-500/20",
  },
  queued: {
    icon: <Clock className="w-6 h-6 text-amber-400" />,
    color: "text-amber-400",
    bg: "bg-amber-500/10",
    border: "border-amber-500/20",
  },
  processing: {
    icon: <Cpu className="w-6 h-6 animate-pulse text-violet-400" />,
    color: "text-violet-400",
    bg: "bg-violet-500/10",
    border: "border-violet-500/20",
  },
  completed: {
    icon: <CheckCircle2 className="w-6 h-6 text-green-400" />,
    color: "text-green-400",
    bg: "bg-green-500/10",
    border: "border-green-500/20",
  },
  failed: {
    icon: <XCircle className="w-6 h-6 text-red-400" />,
    color: "text-red-400",
    bg: "bg-red-500/10",
    border: "border-red-500/20",
  },
};

const pipelineSteps = [
  { id: "receive", label: "Receiving Video", threshold: 5 },
  { id: "extract", label: "Extracting Frames", threshold: 10 },
  { id: "da3", label: "Depth Anything V3 + Gaussian Splatting", threshold: 20 },
  { id: "convert", label: "Converting to Web Format", threshold: 80 },
  { id: "encode", label: "Encoding Result", threshold: 90 },
];

export default function ProcessingStatus({ jobId }: { jobId: string }) {
  const router = useRouter();
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      if (!res.ok) throw new Error("Job not found");
      const data = await res.json();
      setJob(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch status");
    }
  }, [jobId]);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  useEffect(() => {
    if (!job || job.status === "completed" || job.status === "failed") return;
    const timer = setInterval(() => {
      setElapsed((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, [job]);

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  if (error) {
    return (
      <div className="max-w-2xl mx-auto p-8 text-center">
        <XCircle className="w-12 h-12 text-red-400 mx-auto mb-4" />
        <h2 className="text-xl font-bold mb-2">Error</h2>
        <p className="text-zinc-400 mb-6">{error}</p>
        <button
          onClick={() => router.push("/")}
          className="px-6 py-3 bg-zinc-800 hover:bg-zinc-700 rounded-xl transition-colors"
        >
          Back to Upload
        </button>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="max-w-2xl mx-auto p-8 text-center">
        <Loader2 className="w-8 h-8 animate-spin text-violet-400 mx-auto mb-4" />
        <p className="text-zinc-400">Loading job status...</p>
      </div>
    );
  }

  const config = statusConfig[job.status];

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      {/* Status Header */}
      <div
        className={`p-6 rounded-2xl ${config.bg} border ${config.border}`}
      >
        <div className="flex items-center gap-4 mb-4">
          {config.icon}
          <div className="flex-1">
            <h2 className={`text-lg font-bold ${config.color}`}>
              {job.status === "completed"
                ? "Reconstruction Complete!"
                : job.status === "failed"
                ? "Processing Failed"
                : job.status === "queued"
                ? "Waiting in Queue"
                : "Reconstructing 3D Model..."}
            </h2>
            <p className="text-sm text-zinc-400 mt-1">{job.message}</p>
          </div>
          <div className="text-right">
            <p className="text-2xl font-bold tabular-nums">{job.progress}%</p>
            <p className="text-xs text-zinc-500">{formatTime(elapsed)}</p>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="w-full h-3 bg-black/30 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              job.status === "completed"
                ? "bg-gradient-to-r from-green-500 to-emerald-400"
                : job.status === "failed"
                ? "bg-red-500"
                : "bg-gradient-to-r from-violet-600 to-purple-400 progress-pulse"
            }`}
            style={{ width: `${job.progress}%` }}
          />
        </div>
      </div>

      {/* Pipeline Steps */}
      <div className="p-6 rounded-2xl bg-zinc-900 border border-zinc-800">
        <h3 className="text-sm font-semibold text-zinc-400 mb-4 uppercase tracking-wider">
          Pipeline Progress
        </h3>
        <div className="space-y-3">
          {pipelineSteps.map((step) => {
            const isComplete = job.progress >= step.threshold + 5;
            const isActive =
              job.progress >= step.threshold - 5 &&
              job.progress < step.threshold + 5;
            return (
              <div key={step.id} className="flex items-center gap-3">
                <div
                  className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 ${
                    isComplete
                      ? "bg-green-500/20"
                      : isActive
                      ? "bg-violet-500/20"
                      : "bg-zinc-800"
                  }`}
                >
                  {isComplete ? (
                    <CheckCircle2 className="w-4 h-4 text-green-400" />
                  ) : isActive ? (
                    <Loader2 className="w-4 h-4 animate-spin text-violet-400" />
                  ) : (
                    <div className="w-2 h-2 rounded-full bg-zinc-600" />
                  )}
                </div>
                <span
                  className={`text-sm ${
                    isComplete
                      ? "text-green-400"
                      : isActive
                      ? "text-violet-300 font-medium"
                      : "text-zinc-600"
                  }`}
                >
                  {step.label}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Job Info */}
      <div className="p-4 rounded-xl bg-zinc-900/50 border border-zinc-800/50">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-xs text-zinc-500">Quality</p>
            <p className="text-sm font-medium capitalize">{job.quality}</p>
          </div>
          <div>
            <p className="text-xs text-zinc-500">Job ID</p>
            <p className="text-sm font-mono text-zinc-400">
              {job.id.slice(0, 8)}
            </p>
          </div>
          <div>
            <p className="text-xs text-zinc-500">Video</p>
            <p className="text-sm truncate">{job.videoFilename}</p>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      {job.status === "completed" && (
        <button
          onClick={() => router.push(`/viewer/${job.id}`)}
          className="w-full py-4 px-6 rounded-xl font-semibold bg-gradient-to-r from-green-600 to-emerald-500 hover:from-green-500 hover:to-emerald-400 text-white flex items-center justify-center gap-3 shadow-lg shadow-green-600/20 transition-all"
        >
          <Eye className="w-5 h-5" />
          View 3D Model
          <ArrowRight className="w-5 h-5" />
        </button>
      )}

      {job.status === "failed" && (
        <button
          onClick={() => router.push("/")}
          className="w-full py-4 px-6 rounded-xl font-semibold bg-zinc-800 hover:bg-zinc-700 text-white flex items-center justify-center gap-3 transition-all"
        >
          <RotateCcw className="w-5 h-5" />
          Try Again
        </button>
      )}
    </div>
  );
}
