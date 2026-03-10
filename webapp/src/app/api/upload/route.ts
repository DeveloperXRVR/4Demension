import { NextRequest, NextResponse } from "next/server";
import { v4 as uuidv4 } from "uuid";
import { createJob, updateJob } from "@/lib/jobs";
import { submitJob } from "@/lib/runpod";
import { execFile } from "child_process";
import { writeFile, readFile, unlink } from "fs/promises";
import { join } from "path";
import { tmpdir } from "os";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("video") as File | null;
    const quality = (formData.get("quality") as string) || "balanced";
    const maxFrames = parseInt(formData.get("maxFrames") as string) || 200;
    const densityFactor = parseFloat(formData.get("density_factor") as string) || 2.0;
    const generateMesh = formData.get("generate_mesh") === "true";
    const meshMethod = (formData.get("mesh_method") as string) || "poisson";
    const exportFormats = JSON.parse((formData.get("export_formats") as string) || '["splat"]');

    if (!file) {
      return NextResponse.json({ error: "No video file provided" }, { status: 400 });
    }

    const jobId = uuidv4();
    const ext = file.name.split(".").pop() || "mp4";
    const buffer = Buffer.from(await file.arrayBuffer());
    const rawMB = (buffer.length / 1024 / 1024).toFixed(1);
    console.log(`[upload] Received ${rawMB} MB video (${ext})`);
    console.log(`[upload] Settings: density=${densityFactor}x, mesh=${generateMesh}, method=${meshMethod}, formats=${exportFormats.join(",")}`);

    const job = createJob({
      id: jobId,
      status: "uploading",
      videoFilename: file.name,
      videoUrl: "",
      quality: quality as "fast" | "balanced" | "ultra",
      maxFrames,
      progress: 0,
      message: "Compressing video for transfer...",
    });

    const apiKey = process.env.RUNPOD_API_KEY;
    const endpointId = process.env.RUNPOD_ENDPOINT_ID;
    console.log(`[upload] RUNPOD_API_KEY=${apiKey ? apiKey.slice(0, 10) + "..." : "MISSING"}, ENDPOINT=${endpointId || "MISSING"}`);

    if (apiKey && endpointId) {
      submitVideoInBackground(jobId, buffer, ext, quality, maxFrames, densityFactor, generateMesh, meshMethod, exportFormats);
    } else {
      job.status = "queued";
      job.message = "Demo mode: Configure RunPod credentials for GPU processing.";
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

/**
 * Compress video with ffmpeg, base64 encode, send as video_data to RunPod.
 * ffmpeg shrinks 3.3MB → ~200KB, making the RunPod upload fast and reliable.
 */
function submitVideoInBackground(
  jobId: string, buffer: Buffer, ext: string,
  quality: string, maxFrames: number, densityFactor: number,
  generateMesh: boolean, meshMethod: string, exportFormats: string[]
) {
  (async () => {
    try {
      const rawMB = (buffer.length / 1024 / 1024).toFixed(1);

      // Step 1: Compress with ffmpeg
      updateJob(jobId, { message: `Compressing video (${rawMB} MB)...` });
      let videoBuffer: Buffer;
      let videoExt = ext;
      try {
        videoBuffer = await compressVideo(buffer, ext);
        videoExt = "mp4";
        const compMB = (videoBuffer.length / 1024 / 1024).toFixed(2);
        console.log(`[upload] Compressed ${rawMB} MB → ${compMB} MB`);
        updateJob(jobId, { message: `Compressed to ${compMB} MB, uploading to GPU...` });
      } catch (e) {
        console.log(`[upload] ffmpeg compression failed, using raw: ${e instanceof Error ? e.message : e}`);
        videoBuffer = buffer;
      }

      // Step 2: Check payload size and compress if needed
      let videoBase64 = videoBuffer.toString("base64");
      let payloadMB = parseFloat((videoBase64.length / 1024 / 1024).toFixed(2));
      
      // If payload is too large, compress more aggressively
      if (payloadMB > 8) {  // Leave 2MB margin for other parameters
        console.log(`[upload] Payload ${payloadMB}MB too large, compressing more...`);
        updateJob(jobId, { message: "Compressing more to fit size limits..." });
        
        try {
          // Compress with lower bitrate and resolution
          videoBuffer = await compressVideoAggressive(buffer, ext);
          videoBase64 = videoBuffer.toString("base64");
          payloadMB = parseFloat((videoBase64.length / 1024 / 1024).toFixed(2));
          console.log(`[upload] Aggressive compression: ${payloadMB} MB`);
          updateJob(jobId, { message: `Aggressively compressed to ${payloadMB} MB...` });
        } catch (e) {
          console.log(`[upload] Aggressive compression failed: ${e}`);
          // If still too large, reduce frames
          if (payloadMB > 8) {
            const reducedFrames = Math.floor(maxFrames * 0.5);  // Reduce by 50%
            console.log(`[upload] Reducing frames from ${maxFrames} to ${reducedFrames}`);
            maxFrames = reducedFrames;
            updateJob(jobId, { message: `Reducing frames to fit size limits...` });
          }
        }
      }
      
      console.log(`[upload] Final payload: ${payloadMB} MB`);
      updateJob(jobId, { message: `Uploading to GPU cluster (${payloadMB} MB)...` });

      const runpodResponse = await submitJob({
        video_data: videoBase64,
        video_ext: videoExt,
        quality: quality as "fast" | "balanced" | "ultra",
        max_frames: maxFrames,
        density_factor: densityFactor,
        generate_mesh: generateMesh,
        mesh_method: meshMethod,
        export_formats: exportFormats,
      });

      updateJob(jobId, {
        runpodId: runpodResponse.id,
        status: "queued",
        message: "Job submitted to GPU cluster, waiting in queue...",
      });
      console.log(`[upload] RunPod job submitted: ${runpodResponse.id}`);
    } catch (err: unknown) {
      let errMsg = "Unknown error";
      if (err instanceof Error) {
        errMsg = err.message;
        if (err.cause) errMsg += ` (cause: ${err.cause instanceof Error ? err.cause.message : String(err.cause)})`;
      } else {
        errMsg = String(err);
      }
      console.error("RunPod submit error:", errMsg);
      updateJob(jobId, { status: "failed", message: `RunPod submission failed: ${errMsg}` });
    }
  })();
}

/**
 * Compress video with ffmpeg: 320p, CRF 32, no audio, ultrafast.
 * Typical result: 3.3MB → ~150-300KB.
 */
async function compressVideo(buffer: Buffer, ext: string): Promise<Buffer> {
  const inputPath = join(tmpdir(), `4d_in_${Date.now()}.${ext}`);
  const outputPath = join(tmpdir(), `4d_out_${Date.now()}.mp4`);
  await writeFile(inputPath, buffer);

  try {
    await new Promise<void>((resolve, reject) => {
      execFile("ffmpeg", [
        "-y", "-i", inputPath,
        "-vf", "scale=-2:320",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "32",
        "-an",
        "-movflags", "+faststart",
        "-t", "30",
        outputPath,
      ], { timeout: 60_000 }, (err, _stdout, stderr) => {
        if (err) {
          console.log(`[ffmpeg] stderr: ${stderr?.slice(-500)}`);
          reject(err);
        } else {
          resolve();
        }
      });
    });
    return await readFile(outputPath);
  } finally {
    await unlink(inputPath).catch(() => {});
    await unlink(outputPath).catch(() => {});
  }
}

/**
 * Aggressive compression for large videos: 240p, CRF 38, max 15 seconds.
 * Used when payload exceeds 8MB limit.
 */
async function compressVideoAggressive(buffer: Buffer, ext: string): Promise<Buffer> {
  const inputPath = join(tmpdir(), `4d_in_aggressive_${Date.now()}.${ext}`);
  const outputPath = join(tmpdir(), `4d_out_aggressive_${Date.now()}.mp4`);
  await writeFile(inputPath, buffer);

  try {
    await new Promise<void>((resolve, reject) => {
      execFile("ffmpeg", [
        "-y", "-i", inputPath,
        "-vf", "scale=-2:240",  // Lower resolution
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "38",  // Higher compression
        "-an",  // No audio
        "-movflags", "+faststart",
        "-t", "15",  // Max 15 seconds
        outputPath,
      ], { timeout: 60_000 }, (err, _stdout, stderr) => {
        if (err) {
          console.log(`[ffmpeg aggressive] stderr: ${stderr?.slice(-500)}`);
          reject(err);
        } else {
          resolve();
        }
      });
    });
    return await readFile(outputPath);
  } finally {
    await unlink(inputPath).catch(() => {});
    await unlink(outputPath).catch(() => {});
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
