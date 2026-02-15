import { promises as fs } from "fs";
import path from "path";

const UPLOAD_DIR = path.join(process.cwd(), "public", "uploads");
const MODELS_DIR = path.join(process.cwd(), "public", "models");

export async function ensureDirs() {
  await fs.mkdir(UPLOAD_DIR, { recursive: true });
  await fs.mkdir(MODELS_DIR, { recursive: true });
}

export async function saveVideo(
  filename: string,
  buffer: Buffer
): Promise<string> {
  await ensureDirs();
  const filePath = path.join(UPLOAD_DIR, filename);
  await fs.writeFile(filePath, buffer);
  return `/uploads/${filename}`;
}

export async function saveModel(
  jobId: string,
  buffer: Buffer,
  ext: string = "splat"
): Promise<string> {
  await ensureDirs();
  const filePath = path.join(MODELS_DIR, `${jobId}.${ext}`);
  await fs.writeFile(filePath, buffer);
  return `/models/${jobId}.${ext}`;
}

export function getModelPath(jobId: string, ext: string = "splat"): string {
  return path.join(MODELS_DIR, `${jobId}.${ext}`);
}

export function getModelUrl(jobId: string, ext: string = "splat"): string {
  return `/models/${jobId}.${ext}`;
}

export async function modelExists(
  jobId: string,
  ext: string = "splat"
): Promise<boolean> {
  try {
    await fs.access(getModelPath(jobId, ext));
    return true;
  } catch {
    return false;
  }
}
