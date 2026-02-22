import { NextRequest, NextResponse } from "next/server";
import { gunzipSync } from "zlib";
import { getJob, updateJob } from "@/lib/jobs";
import { getJobStatus } from "@/lib/runpod";

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ jobId: string }> }
) {
  const { jobId } = await params;
  const job = getJob(jobId);

  if (!job) {
    return NextResponse.json({ error: "Job not found" }, { status: 404 });
  }

  // If job has a RunPod ID and is still processing, poll RunPod
  if (
    job.runpodId &&
    (job.status === "queued" || job.status === "processing")
  ) {
    try {
      const rpStatus = await getJobStatus(job.runpodId);

      switch (rpStatus.status) {
        case "IN_QUEUE":
          updateJob(jobId, {
            status: "queued",
            message: "Waiting in GPU queue...",
          });
          break;
        case "IN_PROGRESS":
          updateJob(jobId, {
            status: "processing",
            progress: rpStatus.output?.progress || job.progress,
            message: rpStatus.output?.message || "Processing on GPU...",
          });
          break;
        case "COMPLETED":
          if (rpStatus.output?.splat_data) {
            // Decode base64 splat data from RunPod and save locally
            // Data may be gzip-compressed (splat_compressed flag)
            try {
              let splatBuffer = Buffer.from(rpStatus.output.splat_data, "base64");
              if (rpStatus.output.splat_compressed) {
                console.log(`Decompressing gzip splat: ${(splatBuffer.length / 1024 / 1024).toFixed(1)} MB compressed`);
                splatBuffer = gunzipSync(splatBuffer);
                console.log(`Decompressed: ${(splatBuffer.length / 1024 / 1024).toFixed(1)} MB`);
              }
              const { saveModel } = await import("@/lib/storage");
              const modelUrl = await saveModel(jobId, splatBuffer, "splat");
              updateJob(jobId, {
                status: "completed",
                progress: 100,
                message: "3D model reconstruction complete!",
                modelUrl,
              });
            } catch (dlErr) {
              console.error("Splat save error:", dlErr);
              updateJob(jobId, {
                status: "failed",
                message: "Failed to save 3D model",
              });
            }
          } else if (rpStatus.output?.model_url) {
            // Fallback: download from URL
            try {
              const modelRes = await fetch(rpStatus.output.model_url);
              const modelBuffer = Buffer.from(await modelRes.arrayBuffer());
              const { saveModel } = await import("@/lib/storage");
              const modelUrl = await saveModel(jobId, modelBuffer, "splat");
              updateJob(jobId, {
                status: "completed",
                progress: 100,
                message: "3D model reconstruction complete!",
                modelUrl,
              });
            } catch (dlErr) {
              console.error("Model download error:", dlErr);
              updateJob(jobId, {
                status: "failed",
                message: "Failed to download 3D model",
              });
            }
          } else if (rpStatus.output?.error) {
            updateJob(jobId, {
              status: "failed",
              message: rpStatus.output.error,
            });
          } else {
            updateJob(jobId, {
              status: "failed",
              message: "Processing completed but no model data returned",
            });
          }
          break;
        case "FAILED":
          updateJob(jobId, {
            status: "failed",
            message: rpStatus.error || "Processing failed on GPU",
          });
          break;
        case "CANCELLED":
          updateJob(jobId, {
            status: "failed",
            message: "Job was cancelled",
          });
          break;
      }
    } catch (err) {
      console.error("RunPod poll error:", err);
    }
  }

  const updatedJob = getJob(jobId);
  return NextResponse.json(updatedJob);
}
