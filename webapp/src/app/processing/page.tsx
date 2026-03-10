"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Navbar from "@/components/Navbar";
import ProcessingStatus from "@/components/ProcessingStatus";
import { Loader2 } from "lucide-react";

function ProcessingContent() {
  const searchParams = useSearchParams();
  const jobId = searchParams.get("id") || "";

  if (!jobId) {
    return (
      <div className="min-h-screen flex flex-col">
        <Navbar />
        <main className="flex-1 flex items-center justify-center">
          <p className="text-zinc-400">Не указан ID задачи</p>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1 pt-24 pb-16 px-4">
        <div className="max-w-3xl mx-auto">
          <h1 className="text-2xl font-bold text-center mb-2">
            3D реконструкция
          </h1>
          <p className="text-sm text-zinc-500 text-center mb-8">
            Ваше видео обрабатывается на GPU конвейере
          </p>
          <ProcessingStatus jobId={jobId} />
        </div>
      </main>
    </div>
  );
}

export default function ProcessingPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex flex-col">
        <Navbar />
        <main className="flex-1 flex items-center justify-center">
          <Loader2 className="w-8 h-8 animate-spin text-violet-400" />
        </main>
      </div>
    }>
      <ProcessingContent />
    </Suspense>
  );
}
