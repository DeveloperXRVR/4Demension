import https from "https";

function getConfig() {
  const apiKey = process.env.RUNPOD_API_KEY || "";
  const endpointId = process.env.RUNPOD_ENDPOINT_ID || "";
  const baseUrl = `https://api.runpod.ai/v2/${endpointId}`;
  return { apiKey, endpointId, baseUrl };
}

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

/**
 * Low-level HTTPS request with chunked writes and backpressure.
 * Handles slow upload connections that break fetch (EPIPE) and curl (exec fail).
 */
function httpsReq(
  method: "GET" | "POST",
  url: string,
  headers: Record<string, string>,
  body: string,
  timeoutMs = 600_000
): Promise<{ status: number; body: string }> {
  return new Promise((resolve, reject) => {
    const urlObj = new URL(url);
    const bodyBuf = body ? Buffer.from(body, "utf-8") : Buffer.alloc(0);

    const reqHeaders: Record<string, string> = { ...headers };
    if (bodyBuf.length > 0) {
      reqHeaders["Content-Length"] = bodyBuf.length.toString();
    }

    const req = https.request({
      hostname: urlObj.hostname,
      port: 443,
      path: urlObj.pathname,
      method,
      headers: reqHeaders,
      timeout: timeoutMs,
    }, (res) => {
      let data = "";
      res.on("data", (chunk: Buffer) => { data += chunk.toString(); });
      res.on("end", () => resolve({ status: res.statusCode ?? 0, body: data }));
    });

    req.on("error", (err) => reject(new Error(`HTTPS ${method} error: ${err.message}`)));
    req.on("timeout", () => { req.destroy(); reject(new Error(`HTTPS ${method} timed out after ${timeoutMs / 1000}s`)); });

    if (bodyBuf.length === 0) {
      req.end();
      return;
    }

    // Write body in 16KB chunks with backpressure to avoid EPIPE on slow connections
    const CHUNK = 16384;
    let offset = 0;
    let bytesSent = 0;
    const totalMB = (bodyBuf.length / 1024 / 1024).toFixed(1);

    function writeChunks() {
      let ok = true;
      while (ok && offset < bodyBuf.length) {
        const end = Math.min(offset + CHUNK, bodyBuf.length);
        const chunk = bodyBuf.subarray(offset, end);
        offset = end;
        bytesSent += chunk.length;
        ok = req.write(chunk);
      }
      if (offset >= bodyBuf.length) {
        console.log(`[runpod] Upload complete: ${totalMB} MB sent`);
        req.end();
      } else {
        const pct = ((bytesSent / bodyBuf.length) * 100).toFixed(0);
        console.log(`[runpod] Uploading... ${pct}% (${(bytesSent/1024/1024).toFixed(1)}/${totalMB} MB)`);
        req.once("drain", writeChunks);
      }
    }

    writeChunks();
  });
}

export async function submitJob(input: {
  frames_data?: string[];
  video_data?: string;
  video_url?: string;
  video_ext?: string;
  quality: "fast" | "balanced" | "ultra";
  max_frames: number;
  density_factor?: number;
  generate_mesh?: boolean;
  mesh_method?: string;
  export_formats?: string[];
}): Promise<{ id: string }> {
  const { apiKey, baseUrl } = getConfig();
  const body = JSON.stringify({ input });
  const sizeMB = (body.length / 1024 / 1024).toFixed(1);
  console.log(`[runpod] POST ${baseUrl}/run (${sizeMB} MB payload via https.request)`);

  const res = await httpsReq("POST", `${baseUrl}/run`, {
    "Content-Type": "application/json",
    Authorization: `Bearer ${apiKey}`,
  }, body, 600_000);

  console.log(`[runpod] Response: HTTP ${res.status}`);

  if (res.status < 200 || res.status >= 300) {
    throw new Error(`RunPod submit failed: HTTP ${res.status} — ${res.body.slice(0, 500)}`);
  }

  return JSON.parse(res.body) as { id: string };
}

export async function getJobStatus(jobId: string): Promise<RunPodJobResponse> {
  const { apiKey, baseUrl } = getConfig();

  const res = await httpsReq("GET", `${baseUrl}/status/${jobId}`, {
    Authorization: `Bearer ${apiKey}`,
  }, "", 30_000);

  if (res.status < 200 || res.status >= 300) {
    throw new Error(`RunPod status failed: ${res.status} ${res.body.slice(0, 500)}`);
  }

  return JSON.parse(res.body) as RunPodJobResponse;
}

export async function cancelJob(jobId: string): Promise<void> {
  const { apiKey, baseUrl } = getConfig();
  await httpsReq("POST", `${baseUrl}/cancel/${jobId}`, {
    Authorization: `Bearer ${apiKey}`,
  }, "", 10_000);
}
