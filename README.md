# 4Demension — Video to 3D Gaussian Splatting

Upload video footage and reconstruct accurate 3D models using **Depth Anything V3** and **3D Gaussian Splatting**, powered by **RunPod** GPU cloud.

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────────────┐
│  Next.js App │────▶│  RunPod Serverless│────▶│  GPU Worker (Docker)    │
│  (Upload +   │◀────│  API              │◀────│  • COLMAP SfM           │
│   3D Viewer) │     │                  │     │  • Depth Anything V3    │
│              │     │                  │     │  • 3D Gaussian Splatting │
└──────────────┘     └──────────────────┘     └─────────────────────────┘
```

### Pipeline
1. **Upload** — Video uploaded via web UI
2. **Frame Extraction** — ffmpeg extracts key frames
3. **COLMAP SfM** — Camera pose estimation via Structure-from-Motion
4. **Depth Anything V3** — Monocular depth estimation for each frame
5. **3D Gaussian Splatting** — Train 3DGS model with depth priors
6. **Export** — Convert to `.splat` format for web viewing
7. **View** — Interactive Three.js viewer with orbit controls

## Project Structure

```
4Demension/
├── webapp/                  # Next.js frontend + API
│   ├── src/
│   │   ├── app/            # Pages & API routes
│   │   ├── components/     # React components
│   │   └── lib/            # RunPod client, storage, jobs
│   └── public/             # Static files, uploaded videos, models
├── runpod-worker/          # GPU worker for RunPod Serverless
│   ├── Dockerfile          # CUDA + COLMAP + Depth Anything + 3DGS
│   ├── handler.py          # RunPod serverless handler
│   └── requirements.txt    # Python dependencies
└── depth-anything-3/       # Cloned reference repo
```

## Quick Start

### 1. Run the Web App (locally)

```bash
cd webapp
npm install
npm run dev
```

Open http://localhost:3000 — the app runs in **demo mode** (simulated processing) until you configure RunPod.

### 2. Set Up RunPod GPU Worker

#### a) Build the Docker image
```bash
cd runpod-worker
docker build -t 4demension-worker .
```

#### b) Push to Docker Hub (or any registry)
```bash
docker tag 4demension-worker your-dockerhub-username/4demension-worker:latest
docker push your-dockerhub-username/4demension-worker:latest
```

#### c) Create RunPod Serverless Endpoint
1. Go to https://www.runpod.io/console/serverless
2. Click **New Endpoint**
3. Set Docker image: `your-dockerhub-username/4demension-worker:latest`
4. Select GPU: **RTX A5000** or better (24GB+ VRAM recommended)
5. Set Max Workers, Idle Timeout, etc.
6. Copy the **Endpoint ID**

#### d) Configure the Web App
Edit `webapp/.env.local`:
```
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_ID=your_endpoint_id
```

Get your API key from: https://www.runpod.io/console/user/settings

### 3. Use It
1. Open http://localhost:3000
2. Upload a video (walk around an object/scene)
3. Choose quality preset (Fast / Balanced / Ultra)
4. Wait for GPU processing
5. View and interact with your 3D model!

## Tips for Best Results

- **Walk slowly** in a smooth circle around the subject
- **Overlap** — ensure ~60-70% overlap between consecutive frames
- **Lighting** — consistent, diffuse lighting works best
- **Avoid** motion blur, fast movements, transparent/reflective surfaces
- **Duration** — 30-90 seconds of video is ideal

## Tech Stack

- **Frontend**: Next.js 16, React 19, Tailwind CSS v4, Three.js
- **3D Viewer**: Custom Gaussian Splat renderer with Three.js + WebGL shaders
- **Depth Estimation**: [Depth Anything V3](https://github.com/ByteDance-Seed/depth-anything-3) (ViT-L)
- **3D Reconstruction**: [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- **Camera Poses**: COLMAP Structure-from-Motion
- **GPU Cloud**: RunPod Serverless
- **Container**: Docker with NVIDIA CUDA 12.1

## License

This project uses open-source components under their respective licenses.
