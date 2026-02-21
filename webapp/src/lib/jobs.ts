export interface Job {
  id: string;
  runpodId?: string;
  status: "uploading" | "queued" | "processing" | "completed" | "failed";
  videoFilename: string;
  videoUrl: string;
  quality: "fast" | "balanced" | "ultra";
  maxFrames: number;
  modelUrl?: string;
  previewUrl?: string;
  progress: number;
  message: string;
  createdAt: number;
  updatedAt: number;
}

// In-memory job store that survives HMR in dev mode
const globalForJobs = globalThis as unknown as { __jobs?: Map<string, Job> };
if (!globalForJobs.__jobs) {
  globalForJobs.__jobs = new Map<string, Job>();
}
const jobs = globalForJobs.__jobs;

export function createJob(data: Omit<Job, "createdAt" | "updatedAt">): Job {
  const now = Date.now();
  const job: Job = { ...data, createdAt: now, updatedAt: now };
  jobs.set(job.id, job);
  return job;
}

export function getJob(id: string): Job | undefined {
  return jobs.get(id);
}

export function updateJob(id: string, updates: Partial<Job>): Job | undefined {
  const job = jobs.get(id);
  if (!job) return undefined;
  const updated = { ...job, ...updates, updatedAt: Date.now() };
  jobs.set(id, updated);
  return updated;
}

export function getAllJobs(): Job[] {
  return Array.from(jobs.values()).sort((a, b) => b.createdAt - a.createdAt);
}

export function deleteJob(id: string): boolean {
  return jobs.delete(id);
}
