"""
RunPod Serverless Handler for Video → 3D Gaussian Splatting Pipeline.

Simplified pipeline using Depth Anything V3's built-in Gaussian Splatting:
1. Receive video (base64 or URL)
2. Extract frames with ffmpeg
3. Run DA3 with infer_gs=True → depth + camera poses + Gaussian Splats in one pass
4. Export .ply via DA3's built-in gs_ply export
5. Convert to .splat for web viewer
6. Return result as base64
"""

import os
import sys
import time
import shutil
import subprocess
import tempfile
import base64
import struct
import requests
import runpod
import numpy as np

WORKSPACE = "/workspace"


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


def extract_frames(video_path: str, output_dir: str, max_frames: int = 100) -> int:
    """Extract frames from video using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)

    # Get video info
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,duration",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    info = result.stdout.strip().split(",")

    total_frames_est = 300
    try:
        fps_str = info[0]
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
        if len(info) >= 2 and info[1] != "N/A":
            duration = float(info[1])
            total_frames_est = int(fps * duration)
    except (ValueError, ZeroDivisionError, IndexError):
        pass

    interval = max(1, total_frames_est // max_frames)

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


def run_da3_gaussian_splatting(images_dir: str, export_dir: str) -> str:
    """Run DA3 with built-in Gaussian Splatting to generate PLY directly."""
    import torch
    from depth_anything_3.api import DepthAnything3

    print("Loading Depth Anything V3 (DA3-LARGE)...")
    model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Collect image paths
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    image_paths = [os.path.join(images_dir, f) for f in image_files]
    print(f"Running DA3 Gaussian Splatting on {len(image_paths)} frames...")

    # Run DA3 with Gaussian Splatting enabled — single pass does everything
    prediction = model.inference(
        image=image_paths,
        infer_gs=True,
        export_dir=export_dir,
        export_format="gs_ply",
    )

    print(f"DA3 inference complete. Depth shape: {prediction.depth.shape}")
    if prediction.gaussians is not None:
        print(f"Gaussians generated: {prediction.gaussians.means.shape[0]} splats")

    del model
    torch.cuda.empty_cache()

    # Find the exported PLY
    ply_path = os.path.join(export_dir, "gs_ply", "0000.ply")
    if not os.path.exists(ply_path):
        # Search for any PLY in export dir
        for root, dirs, files in os.walk(export_dir):
            for f in files:
                if f.endswith(".ply"):
                    ply_path = os.path.join(root, f)
                    print(f"Found PLY at: {ply_path}")
                    break

    if not os.path.exists(ply_path):
        raise RuntimeError(f"DA3 did not produce a PLY file in {export_dir}")

    print(f"PLY exported: {ply_path} ({os.path.getsize(ply_path) / 1024 / 1024:.1f} MB)")
    return ply_path


def convert_ply_to_splat(ply_path: str, splat_path: str) -> str:
    """Convert PLY point cloud to .splat format for web viewer."""
    from plyfile import PlyData

    print("Converting PLY to .splat format...")
    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]
    num_points = len(vertex)
    print(f"Converting {num_points} Gaussians to .splat")

    # Extract properties - handle different PLY formats
    prop_names = [p.name for p in vertex.properties]

    # Positions
    positions = np.column_stack([vertex["x"], vertex["y"], vertex["z"]])

    # Scales
    if "scale_0" in prop_names:
        scales = np.column_stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]])
        scales = np.exp(scales)
    else:
        scales = np.ones((num_points, 3), dtype=np.float32) * 0.01

    # Rotations (quaternion)
    if "rot_0" in prop_names:
        rotations = np.column_stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]])
    elif "qw" in prop_names:
        rotations = np.column_stack([vertex["qw"], vertex["qx"], vertex["qy"], vertex["qz"]])
    else:
        rotations = np.tile([1, 0, 0, 0], (num_points, 1)).astype(np.float32)
    norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    rotations = rotations / (norms + 1e-10)

    # Colors
    if "f_dc_0" in prop_names:
        sh_dc = np.column_stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]])
        C0 = 0.28209479177387814
        colors = (sh_dc * C0 + 0.5).clip(0, 1)
    elif "red" in prop_names:
        colors = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]).astype(np.float32)
        if colors.max() > 1.0:
            colors = colors / 255.0
    else:
        colors = np.ones((num_points, 3), dtype=np.float32) * 0.5

    # Opacity
    if "opacity" in prop_names:
        raw_opacity = np.array(vertex["opacity"], dtype=np.float32)
        if raw_opacity.min() < 0:
            opacity = 1.0 / (1.0 + np.exp(-raw_opacity))
        else:
            opacity = raw_opacity.clip(0, 1)
    else:
        opacity = np.ones(num_points, dtype=np.float32)

    # Sort by scale for better rendering
    scale_sum = scales.sum(axis=1)
    sort_idx = np.argsort(-scale_sum)
    positions = positions[sort_idx]
    scales = scales[sort_idx]
    rotations = rotations[sort_idx]
    colors = colors[sort_idx]
    opacity = opacity[sort_idx]

    # Write .splat binary format: pos(3f) + scale(3f) + rgba(4B) + rot(4B) = 32 bytes
    buf = bytearray(num_points * 32)
    for i in range(num_points):
        off = i * 32
        struct.pack_into('3f', buf, off, *positions[i])
        struct.pack_into('3f', buf, off + 12, *scales[i])
        buf[off + 24] = min(255, max(0, int(colors[i, 0] * 255)))
        buf[off + 25] = min(255, max(0, int(colors[i, 1] * 255)))
        buf[off + 26] = min(255, max(0, int(colors[i, 2] * 255)))
        buf[off + 27] = min(255, max(0, int(opacity[i] * 255)))
        struct.pack_into('4B', buf, off + 28,
            min(255, max(0, int((rotations[i, 0] * 0.5 + 0.5) * 255))),
            min(255, max(0, int((rotations[i, 1] * 0.5 + 0.5) * 255))),
            min(255, max(0, int((rotations[i, 2] * 0.5 + 0.5) * 255))),
            min(255, max(0, int((rotations[i, 3] * 0.5 + 0.5) * 255))),
        )

    with open(splat_path, "wb") as f:
        f.write(buf)

    file_size_mb = os.path.getsize(splat_path) / (1024 * 1024)
    print(f"Splat file: {splat_path} ({file_size_mb:.1f} MB, {num_points} points)")
    return splat_path


def handler(event):
    """RunPod serverless handler."""
    job_input = event["input"]
    job_id = event.get("id", "unknown")

    max_frames = job_input.get("max_frames", 100)
    video_ext = job_input.get("video_ext", "mp4")

    work_dir = tempfile.mkdtemp(prefix="4d_", dir=WORKSPACE)
    images_dir = os.path.join(work_dir, "images")
    export_dir = os.path.join(work_dir, "export")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)

    try:
        # Step 1: Get video
        runpod.serverless.progress_update(event, {"progress": 5, "message": "Receiving video..."})
        video_path = os.path.join(work_dir, f"input.{video_ext}")

        if "video_data" in job_input:
            print("Decoding base64 video data...")
            video_bytes = base64.b64decode(job_input["video_data"])
            with open(video_path, "wb") as f:
                f.write(video_bytes)
            print(f"Video decoded: {len(video_bytes)} bytes")
        elif "video_url" in job_input:
            download_video(job_input["video_url"], video_path)
        else:
            return {"error": "No video_data or video_url provided"}

        # Step 2: Extract frames
        runpod.serverless.progress_update(event, {"progress": 10, "message": "Extracting frames from video..."})
        num_frames = extract_frames(video_path, images_dir, max_frames)

        if num_frames < 2:
            return {"error": f"Only {num_frames} frame(s) extracted. Need at least 2. Try a longer video."}

        # Step 3: DA3 — depth + poses + Gaussian Splats in one pass
        runpod.serverless.progress_update(event, {"progress": 20, "message": "Running Depth Anything V3 + Gaussian Splatting..."})
        ply_path = run_da3_gaussian_splatting(images_dir, export_dir)

        # Step 4: Convert to .splat for web viewer
        runpod.serverless.progress_update(event, {"progress": 80, "message": "Converting to web format..."})
        splat_path = os.path.join(work_dir, f"{job_id}.splat")
        convert_ply_to_splat(ply_path, splat_path)

        # Step 5: Return result as base64 (no cloud storage needed)
        runpod.serverless.progress_update(event, {"progress": 90, "message": "Encoding result..."})
        with open(splat_path, "rb") as f:
            splat_data = base64.b64encode(f.read()).decode("utf-8")

        file_size_mb = os.path.getsize(splat_path) / (1024 * 1024)
        print(f"Returning splat: {file_size_mb:.1f} MB, {num_frames} frames processed")

        return {
            "splat_data": splat_data,
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
        import traceback
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}

    finally:
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    print("Pre-downloading Depth Anything V3 model weights...")
    try:
        from depth_anything_3.api import DepthAnything3
        _model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE")
        del _model
        print("Model weights cached successfully.")
    except Exception as e:
        print(f"WARNING: Could not pre-download model: {e}")
        print("Model will be downloaded on first job.")

    runpod.serverless.start({"handler": handler})
