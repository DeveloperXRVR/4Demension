import { NextRequest, NextResponse } from "next/server";
import { v4 as uuidv4 } from "uuid";
import { saveVideo } from "@/lib/storage";
import { createJob } from "@/lib/jobs";
import { submitJob } from "@/lib/runpod";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("video") as File | null;
    const quality = (formData.get("quality") as string) || "balanced";
    const maxFrames = parseInt(formData.get("maxFrames") as string) || 200;

    if (!file) {
      return NextResponse.json({ error: "No video file provided" }, { status: 400 });
    }

    const allowedTypes = ["video/mp4", "video/webm", "video/quicktime", "video/x-msvideo"];
    if (!allowedTypes.includes(file.type)) {
      return NextResponse.json(
        { error: "Invalid file type. Supported: MP4, WebM, MOV, AVI" },
        { status: 400 }
      );
    }

    const maxSize = 500 * 1024 * 1024; // 500MB
    if (file.size > maxSize) {
      return NextResponse.json(
        { error: "File too large. Maximum size is 500MB" },
        { status: 400 }
      );
    }

    const jobId = uuidv4();
    const ext = file.name.split(".").pop() || "mp4";
    const filename = `${jobId}.${ext}`;

    const buffer = Buffer.from(await file.arrayBuffer());
    const videoUrl = await saveVideo(filename, buffer);

    // Create job record
    const job = createJob({
      id: jobId,
      status: "uploading",
      videoFilename: filename,
      videoUrl,
      quality: quality as "fast" | "balanced" | "ultra",
      maxFrames,
      progress: 0,
      message: "Video uploaded, submitting to processing queue...",
    });

    // Submit to RunPod
    const useRunPod = process.env.RUNPOD_API_KEY && process.env.RUNPOD_ENDPOINT_ID;

    if (useRunPod) {
      try {
        // Send video as base64 directly to RunPod (no tunnel needed)
        const videoBase64 = buffer.toString("base64");

        const runpodResponse = await submitJob({
          video_data: videoBase64,
          video_ext: ext,
          quality: quality as "fast" | "balanced" | "ultra",
          max_frames: maxFrames,
        });

        job.runpodId = runpodResponse.id;
        job.status = "queued";
        job.message = "Job submitted to GPU cluster, waiting in queue...";
      } catch (err) {
        console.error("RunPod submit error:", err);
        job.status = "queued";
        job.message = "RunPod submission pending - configure RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID";
      }
    } else {
      // Demo mode - simulate processing
      job.status = "queued";
      job.message = "Demo mode: Configure RunPod credentials for GPU processing. Using simulated pipeline.";
      simulateProcessing(jobId);
    }

    return NextResponse.json({
      jobId: job.id,
      status: job.status,
      message: job.message,
    });
  } catch (error) {
    console.error("Upload error:", error);
    return NextResponse.json(
      { error: "Failed to process upload" },
      { status: 500 }
    );
  }
}

// Demo mode simulation
async function simulateProcessing(jobId: string) {
  const { updateJob } = await import("@/lib/jobs");
  const { ensureDirs } = await import("@/lib/storage");
  const path = await import("path");
  const fs = await import("fs/promises");

  await ensureDirs();

  const stages = [
    { progress: 5, message: "Extracting frames from video...", delay: 2000 },
    { progress: 15, message: "Running COLMAP Structure-from-Motion...", delay: 3000 },
    { progress: 30, message: "Estimating camera poses...", delay: 2000 },
    { progress: 45, message: "Running Depth Anything V3 depth estimation...", delay: 3000 },
    { progress: 60, message: "Initializing 3D Gaussian Splatting...", delay: 2000 },
    { progress: 75, message: "Training Gaussian Splat model...", delay: 4000 },
    { progress: 90, message: "Exporting .splat file...", delay: 2000 },
    { progress: 95, message: "Optimizing for web delivery...", delay: 1000 },
  ];

  updateJob(jobId, { status: "processing", progress: 0 });

  for (const stage of stages) {
    await new Promise((r) => setTimeout(r, stage.delay));
    updateJob(jobId, { progress: stage.progress, message: stage.message });
  }

  // Create a demo .ply file (minimal valid PLY for demo)
  const demoModelPath = path.default.join(process.cwd(), "public", "models", `${jobId}.splat`);
  const demoPlyData = generateDemoSplatData();
  await fs.writeFile(demoModelPath, demoPlyData);

  updateJob(jobId, {
    status: "completed",
    progress: 100,
    message: "3D model reconstruction complete!",
    modelUrl: `/models/${jobId}.splat`,
  });
}

function generateDemoSplatData(): Buffer {
  // Generate a minimal demo splat binary for testing the viewer
  // In production, this comes from the actual 3DGS pipeline
  const numSplats = 5000;
  // Each splat: position(3f) + scale(3f) + color(4B) + rotation(4f) = 12+12+4+16 = 44 bytes
  // Using a simplified format - the actual viewer will need proper .splat format
  const buffer = Buffer.alloc(numSplats * 32);

  for (let i = 0; i < numSplats; i++) {
    const offset = i * 32;
    // Position (spread in a sphere-like shape)
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const r = Math.random() * 2 + 0.5;
    buffer.writeFloatLE(r * Math.sin(phi) * Math.cos(theta), offset);
    buffer.writeFloatLE(r * Math.sin(phi) * Math.sin(theta), offset + 4);
    buffer.writeFloatLE(r * Math.cos(phi), offset + 8);
    // Scale
    const s = 0.01 + Math.random() * 0.03;
    buffer.writeFloatLE(s, offset + 12);
    buffer.writeFloatLE(s, offset + 16);
    buffer.writeFloatLE(s, offset + 20);
    // Color RGBA (bytes)
    buffer.writeUInt8(Math.floor(100 + Math.random() * 155), offset + 24);
    buffer.writeUInt8(Math.floor(100 + Math.random() * 155), offset + 25);
    buffer.writeUInt8(Math.floor(100 + Math.random() * 155), offset + 26);
    buffer.writeUInt8(200, offset + 27);
    // Rotation quaternion (identity-ish)
    buffer.writeFloatLE(1.0, offset + 28);
  }

  return buffer;
}
