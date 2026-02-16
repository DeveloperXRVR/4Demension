"""
RunPod Serverless Handler for Video → 3D Gaussian Splatting Pipeline.

Pipeline:
1. Download video from URL
2. Extract frames with ffmpeg
3. Run COLMAP for Structure-from-Motion (camera poses)
4. Run Depth Anything V3 for monocular depth estimation
5. Train 3D Gaussian Splatting with depth priors
6. Export .splat file
7. Upload result and return URL
"""

import os
import sys
import time
import shutil
import subprocess
import tempfile
import base64
import requests
import runpod
import numpy as np

WORKSPACE = "/workspace"
DEPTH_ANYTHING_PATH = "/workspace/depth-anything-3"
GAUSSIAN_SPLATTING_PATH = "/workspace/gaussian-splatting"


def download_video(url: str, output_path: str) -> str:
    """Download video from URL."""
    print(f"Downloading video from {url}")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Video downloaded: {os.path.getsize(output_path)} bytes")
    return output_path


def extract_frames(video_path: str, output_dir: str, max_frames: int = 200) -> int:
    """Extract frames from video using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)

    # Get video duration and fps
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_packets", "-show_entries",
        "stream=nb_read_packets,r_frame_rate,duration",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    info = result.stdout.strip().split(",")

    # Calculate frame interval to get desired number of frames
    total_frames_est = 300  # fallback
    try:
        if len(info) >= 1:
            fps_str = info[0]
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
            if len(info) >= 3 and info[2] != "N/A":
                duration = float(info[2])
                total_frames_est = int(fps * duration)
    except (ValueError, ZeroDivisionError):
        pass

    interval = max(1, total_frames_est // max_frames)

    # Extract frames
    extract_cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"select=not(mod(n\\,{interval}))",
        "-vsync", "vfr",
        "-q:v", "1",
        os.path.join(output_dir, "frame_%06d.jpg")
    ]
    subprocess.run(extract_cmd, capture_output=True, check=True)

    num_frames = len([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
    print(f"Extracted {num_frames} frames (interval={interval})")
    return num_frames


def run_colmap(images_dir: str, workspace_dir: str) -> str:
    """Run COLMAP Structure-from-Motion to estimate camera poses."""
    db_path = os.path.join(workspace_dir, "database.db")
    sparse_dir = os.path.join(workspace_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    # Feature extraction
    print("COLMAP: Feature extraction...")
    subprocess.run([
        "colmap", "feature_extractor",
        "--database_path", db_path,
        "--image_path", images_dir,
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "OPENCV",
        "--SiftExtraction.use_gpu", "1",
    ], check=True, capture_output=True)

    # Feature matching
    print("COLMAP: Feature matching...")
    subprocess.run([
        "colmap", "sequential_matcher",
        "--database_path", db_path,
        "--SequentialMatching.overlap", "10",
        "--SiftMatching.use_gpu", "1",
    ], check=True, capture_output=True)

    # Sparse reconstruction
    print("COLMAP: Sparse reconstruction...")
    subprocess.run([
        "colmap", "mapper",
        "--database_path", db_path,
        "--image_path", images_dir,
        "--output_path", sparse_dir,
    ], check=True, capture_output=True)

    # Undistort images
    dense_dir = os.path.join(workspace_dir, "dense")
    print("COLMAP: Image undistortion...")
    subprocess.run([
        "colmap", "image_undistorter",
        "--image_path", images_dir,
        "--input_path", os.path.join(sparse_dir, "0"),
        "--output_path", dense_dir,
        "--output_type", "COLMAP",
    ], check=True, capture_output=True)

    print("COLMAP: Done")
    return sparse_dir


def run_depth_anything(images_dir: str, output_dir: str) -> str:
    """Run Depth Anything V3 for monocular depth estimation."""
    import torch
    from PIL import Image
    from depth_anything_3 import DepthAnything3

    os.makedirs(output_dir, exist_ok=True)

    print("Loading Depth Anything V3 (DA3-LARGE)...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"Running depth estimation on {len(image_files)} frames...")
    for i, fname in enumerate(image_files):
        img_path = os.path.join(images_dir, fname)
        img = Image.open(img_path).convert("RGB")

        with torch.no_grad():
            depth = model.infer(img)

        # Save depth map as 16-bit PNG
        depth_np = depth.squeeze().cpu().numpy()
        depth_normalized = ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8) * 65535).astype(np.uint16)
        depth_img = Image.fromarray(depth_normalized)
        out_name = os.path.splitext(fname)[0] + ".png"
        depth_img.save(os.path.join(output_dir, out_name))

        if (i + 1) % 20 == 0:
            print(f"  Depth: {i+1}/{len(image_files)} frames done")

    del model
    torch.cuda.empty_cache()

    num_depths = len([f for f in os.listdir(output_dir) if f.endswith(".png")])
    print(f"Generated {num_depths} depth maps")
    return output_dir


def train_gaussian_splatting(
    colmap_dir: str,
    depth_dir: str,
    output_dir: str,
    quality: str = "balanced"
) -> str:
    """Train 3D Gaussian Splatting model with depth priors."""
    os.makedirs(output_dir, exist_ok=True)

    # Quality presets
    iterations = {"fast": 7000, "balanced": 15000, "ultra": 30000}
    num_iter = iterations.get(quality, 15000)

    print(f"Training 3D Gaussian Splatting ({quality}: {num_iter} iterations)...")
    subprocess.run([
        sys.executable,
        os.path.join(GAUSSIAN_SPLATTING_PATH, "train.py"),
        "-s", colmap_dir,
        "-m", output_dir,
        "--iterations", str(num_iter),
        "--depth_regularization",
        "--depth_path", depth_dir,
        "--save_iterations", str(num_iter),
    ], check=True, capture_output=True, cwd=GAUSSIAN_SPLATTING_PATH)

    # Find the output PLY
    ply_path = os.path.join(output_dir, "point_cloud", f"iteration_{num_iter}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        # Try to find any PLY
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.endswith(".ply"):
                    ply_path = os.path.join(root, f)
                    break

    print(f"3DGS training complete: {ply_path}")
    return ply_path


def convert_ply_to_splat(ply_path: str, splat_path: str) -> str:
    """Convert PLY point cloud to .splat format for web viewer."""
    from plyfile import PlyData

    print("Converting PLY to .splat format...")
    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]

    num_points = len(vertex)
    print(f"Converting {num_points} Gaussians to .splat")

    # Extract data
    positions = np.column_stack([vertex["x"], vertex["y"], vertex["z"]])

    # Extract scales (log scale)
    scales = np.column_stack([
        vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]
    ])
    scales = np.exp(scales)

    # Extract rotations (quaternion)
    rotations = np.column_stack([
        vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]
    ])
    # Normalize quaternions
    norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    rotations = rotations / (norms + 1e-10)

    # Extract spherical harmonics (just DC component for color)
    sh_dc = np.column_stack([
        vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]
    ])
    # SH to RGB: C0 * sh + 0.5
    C0 = 0.28209479177387814
    colors = (sh_dc * C0 + 0.5).clip(0, 1)

    # Opacity (sigmoid)
    opacity = 1.0 / (1.0 + np.exp(-vertex["opacity"]))

    # Sort by scale for better rendering
    scale_sum = scales.sum(axis=1)
    sort_idx = np.argsort(-scale_sum)

    positions = positions[sort_idx]
    scales = scales[sort_idx]
    rotations = rotations[sort_idx]
    colors = colors[sort_idx]
    opacity = opacity[sort_idx]

    # Write .splat binary format
    # Format per splat: pos(3f) + scale(3f) + rgba(4B) + rot(4B) = 32 bytes
    buffer = bytearray(num_points * 32)

    for i in range(num_points):
        offset = i * 32
        # Position (3 x float32 = 12 bytes)
        pos_bytes = np.array(positions[i], dtype=np.float32).tobytes()
        buffer[offset:offset+12] = pos_bytes

        # Scale (3 x float32 = 12 bytes)
        sc_bytes = np.array(scales[i], dtype=np.float32).tobytes()
        buffer[offset+12:offset+24] = sc_bytes

        # Color RGBA (4 bytes)
        buffer[offset+24] = int(colors[i, 0] * 255)
        buffer[offset+25] = int(colors[i, 1] * 255)
        buffer[offset+26] = int(colors[i, 2] * 255)
        buffer[offset+27] = int(opacity[i] * 255)

        # Rotation quaternion (normalized to uint8 range, or first float)
        rot_bytes = np.array([rotations[i, 0]], dtype=np.float32).tobytes()
        buffer[offset+28:offset+32] = rot_bytes

    with open(splat_path, "wb") as f:
        f.write(buffer)

    file_size_mb = os.path.getsize(splat_path) / (1024 * 1024)
    print(f"Splat file: {splat_path} ({file_size_mb:.1f} MB, {num_points} points)")
    return splat_path


def upload_result(file_path: str, job_id: str) -> str:
    """Upload the result file to cloud storage. Returns download URL."""
    # Option 1: S3 upload
    s3_bucket = os.environ.get("S3_BUCKET")
    if s3_bucket:
        import boto3
        s3 = boto3.client("s3")
        key = f"models/{job_id}.splat"
        s3.upload_file(file_path, s3_bucket, key)
        region = os.environ.get("AWS_REGION", "us-east-1")
        return f"https://{s3_bucket}.s3.{region}.amazonaws.com/{key}"

    # Option 2: RunPod network volume (return local path)
    network_vol = "/runpod-volume"
    if os.path.exists(network_vol):
        output_path = os.path.join(network_vol, "models", f"{job_id}.splat")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(file_path, output_path)
        return output_path

    # Fallback: serve from temp (will be lost when pod stops)
    return file_path


def handler(event):
    """RunPod serverless handler."""
    job_input = event["input"]
    job_id = event.get("id", "unknown")

    quality = job_input.get("quality", "balanced")
    max_frames = job_input.get("max_frames", 200)
    video_ext = job_input.get("video_ext", "mp4")

    # Create working directory
    work_dir = tempfile.mkdtemp(prefix="4d_", dir=WORKSPACE)
    images_dir = os.path.join(work_dir, "images")
    depth_dir = os.path.join(work_dir, "depths")
    colmap_dir = os.path.join(work_dir, "colmap")
    model_dir = os.path.join(work_dir, "model")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(colmap_dir, exist_ok=True)

    try:
        # Step 1: Get video (base64 or URL)
        runpod.serverless.progress_update(event, {"progress": 5, "message": "Receiving video..."})
        video_path = os.path.join(work_dir, f"input.{video_ext}")

        if "video_data" in job_input:
            # Decode base64 video data
            print("Decoding base64 video data...")
            video_bytes = base64.b64decode(job_input["video_data"])
            with open(video_path, "wb") as f:
                f.write(video_bytes)
            print(f"Video decoded: {len(video_bytes)} bytes")
        elif "video_url" in job_input:
            # Fallback: download from URL
            download_video(job_input["video_url"], video_path)
        else:
            return {"error": "No video_data or video_url provided"}

        # Step 2: Extract frames
        runpod.serverless.progress_update(event, {"progress": 10, "message": "Extracting frames from video..."})
        num_frames = extract_frames(video_path, images_dir, max_frames)

        # Step 3: COLMAP SfM
        runpod.serverless.progress_update(event, {"progress": 20, "message": "Running COLMAP Structure-from-Motion..."})
        sparse_dir = run_colmap(images_dir, colmap_dir)

        # Step 4: Depth estimation
        runpod.serverless.progress_update(event, {"progress": 45, "message": "Running Depth Anything V3 depth estimation..."})
        run_depth_anything(images_dir, depth_dir)

        # Step 5: Train 3DGS
        runpod.serverless.progress_update(event, {"progress": 55, "message": "Training 3D Gaussian Splatting..."})
        ply_path = train_gaussian_splatting(
            os.path.join(colmap_dir, "dense"),
            depth_dir,
            model_dir,
            quality
        )

        # Step 6: Convert to .splat
        runpod.serverless.progress_update(event, {"progress": 90, "message": "Converting to web format..."})
        splat_path = os.path.join(work_dir, f"{job_id}.splat")
        convert_ply_to_splat(ply_path, splat_path)

        # Step 7: Upload result
        runpod.serverless.progress_update(event, {"progress": 95, "message": "Uploading model..."})
        model_url = upload_result(splat_path, job_id)

        return {
            "model_url": model_url,
            "num_frames": num_frames,
            "progress": 100,
            "message": "3D reconstruction complete!",
        }

    except subprocess.CalledProcessError as e:
        error_msg = f"Pipeline step failed: {e.cmd[0] if e.cmd else 'unknown'}"
        if e.stderr:
            error_msg += f" — {e.stderr[:500]}"
        print(f"ERROR: {error_msg}")
        return {"error": error_msg}

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {"error": str(e)}

    finally:
        # Cleanup working directory
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    # Pre-download model weights on worker startup (not per-job)
    print("Pre-downloading Depth Anything V3 model weights...")
    try:
        from depth_anything_3 import DepthAnything3
        _model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
        del _model
        print("Model weights cached successfully.")
    except Exception as e:
        print(f"WARNING: Could not pre-download model: {e}")
        print("Model will be downloaded on first job.")

    runpod.serverless.start({"handler": handler})
