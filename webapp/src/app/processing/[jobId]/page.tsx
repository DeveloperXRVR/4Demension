"use client";

import { use } from "react";
import Navbar from "@/components/Navbar";
import ProcessingStatus from "@/components/ProcessingStatus";

export default function ProcessingPage({
  params,
}: {
  params: Promise<{ jobId: string }>;
}) {
  const { jobId } = use(params);

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1 pt-24 pb-16 px-4">
        <div className="max-w-3xl mx-auto">
          <h1 className="text-2xl font-bold text-center mb-2">
            3D Reconstruction
          </h1>
          <p className="text-sm text-zinc-500 text-center mb-8">
            Your video is being processed by the GPU pipeline
          </p>
          <ProcessingStatus jobId={jobId} />
        </div>
      </main>
    </div>
  );
}
