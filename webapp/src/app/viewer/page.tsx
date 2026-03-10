"use client";

import { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Navbar from "@/components/Navbar";
import GaussianSplatViewer from "@/components/GaussianSplatViewer";
import { ArrowLeft, Loader2, XCircle } from "lucide-react";

interface Job {
  id: string;
  status: string;
  modelUrl?: string;
  quality: string;
  videoFilename: string;
}

function ViewerContent() {
  const searchParams = useSearchParams();
  const jobId = searchParams.get("id") || "";
  const directModel = searchParams.get("model");
  const router = useRouter();
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (directModel) {
      setJob({
        id: jobId || "direct",
        status: "completed",
        modelUrl: directModel,
        quality: "—",
        videoFilename: "—",
      });
      return;
    }

    if (!jobId) {
      setError("Не указан ID задачи");
      return;
    }

    async function fetchJob() {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) throw new Error("Задача не найдена");
        const data = await res.json();
        if (data.status !== "completed" || !data.modelUrl) {
          router.push(`/processing/?id=${jobId}`);
          return;
        }
        setJob(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Не удалось загрузить задачу");
      }
    }
    fetchJob();
  }, [jobId, directModel, router]);

  if (error) {
    return (
      <div className="min-h-screen flex flex-col">
        <Navbar />
        <main className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <XCircle className="w-12 h-12 text-red-400 mx-auto mb-4" />
            <h2 className="text-xl font-bold mb-2">Ошибка</h2>
            <p className="text-zinc-400 mb-6">{error}</p>
            <button
              onClick={() => router.push("/")}
              className="px-6 py-3 bg-zinc-800 hover:bg-zinc-700 rounded-xl transition-colors"
            >
              Назад к загрузке
            </button>
          </div>
        </main>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="min-h-screen flex flex-col">
        <Navbar />
        <main className="flex-1 flex items-center justify-center">
          <Loader2 className="w-8 h-8 animate-spin text-violet-400" />
        </main>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col">
      <Navbar />
      <div className="pt-16 flex-1 flex flex-col min-h-0">
        <div className="flex items-center justify-between px-4 py-2 bg-zinc-950 border-b border-zinc-800">
          <button
            onClick={() => router.push("/")}
            className="flex items-center gap-2 text-sm text-zinc-400 hover:text-white transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Новая загрузка
          </button>
          <div className="flex items-center gap-4 text-xs text-zinc-500">
            <span>
              Качество: <span className="text-zinc-300 capitalize">{job.quality}</span>
            </span>
            <span>
              Задача: <span className="font-mono text-zinc-300">{job.id.slice(0, 8)}</span>
            </span>
          </div>
        </div>
        <div className="flex-1 min-h-0">
          <GaussianSplatViewer modelUrl={job.modelUrl!} jobId={job.id} />
        </div>
      </div>
    </div>
  );
}

export default function ViewerPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex flex-col">
        <Navbar />
        <main className="flex-1 flex items-center justify-center">
          <Loader2 className="w-8 h-8 animate-spin text-violet-400" />
        </main>
      </div>
    }>
      <ViewerContent />
    </Suspense>
  );
}
