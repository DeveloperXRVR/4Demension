const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY || "";
const RUNPOD_ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID || "";
const RUNPOD_BASE_URL = `https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}`;

export interface RunPodJobResponse {
  id: string;
  status: "IN_QUEUE" | "IN_PROGRESS" | "COMPLETED" | "FAILED" | "CANCELLED";
  output?: {
    splat_data?: string;
    splat_compressed?: boolean;
    model_url?: string;
    preview_url?: string;
    num_frames?: number;
    progress?: number;
    message?: string;
    error?: string;
  };
  error?: string;
}

export async function submitJob(input: {
  video_data: string; // base64 encoded video
  video_ext: string;
  quality: "fast" | "balanced" | "ultra";
  max_frames: number;
}): Promise<{ id: string }> {
  const res = await fetch(`${RUNPOD_BASE_URL}/run`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${RUNPOD_API_KEY}`,
    },
    body: JSON.stringify({ input }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`RunPod submit failed: ${res.status} ${err}`);
  }

  return res.json();
}

export async function getJobStatus(jobId: string): Promise<RunPodJobResponse> {
  const res = await fetch(`${RUNPOD_BASE_URL}/status/${jobId}`, {
    headers: {
      Authorization: `Bearer ${RUNPOD_API_KEY}`,
    },
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`RunPod status failed: ${res.status} ${err}`);
  }

  return res.json();
}

export async function cancelJob(jobId: string): Promise<void> {
  await fetch(`${RUNPOD_BASE_URL}/cancel/${jobId}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${RUNPOD_API_KEY}`,
    },
  });
}
