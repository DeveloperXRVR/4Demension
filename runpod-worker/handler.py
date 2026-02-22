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


MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE"


def run_da3_reconstruction(images_dir: str, export_dir: str, max_points: int = 500000) -> str:
    """Run DA3 depth + GS reconstruction.  Tries built-in gs_ply first, falls back to manual point cloud."""
    import torch
    from depth_anything_3.api import DepthAnything3

    # Collect image paths
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    image_paths = [os.path.join(images_dir, f) for f in image_files]
    print(f"Running DA3 on {len(image_paths)} frames...")

    print(f"Loading Depth Anything V3 ({MODEL_ID})...")
    model = DepthAnything3.from_pretrained(MODEL_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"Model loaded on {device}")

    # --- Attempt 1: built-in Gaussian Splatting export ---
    gs_ply_path = os.path.join(export_dir, "gs_ply", "000.ply")
    try:
        print("Running DA3 with infer_gs=True + gs_ply export...")
        prediction = model.inference(
            image=image_paths,
            infer_gs=True,
            export_dir=export_dir,
            export_format="gs_ply",
            ref_view_strategy="middle",
        )
        # DA3 saves gs_ply under export_dir/gs_ply/
        # Find the actual PLY file
        gs_dir = os.path.join(export_dir, "gs_ply")
        if os.path.isdir(gs_dir):
            ply_files = [f for f in os.listdir(gs_dir) if f.endswith(".ply")]
            if ply_files:
                gs_ply_path = os.path.join(gs_dir, ply_files[0])
                size_mb = os.path.getsize(gs_ply_path) / (1024 * 1024)
                print(f"GS PLY exported: {gs_ply_path} ({size_mb:.1f} MB)")
                del model
                torch.cuda.empty_cache()
                return gs_ply_path
        print("GS PLY export dir empty, falling back to manual point cloud")
    except Exception as e:
        print(f"GS export failed ({e}), falling back to manual point cloud")

    # --- Attempt 2: manual point cloud from depth maps ---
    print("Running DA3 depth-only inference...")
    try:
        prediction = model.inference(
            image=image_paths,
            infer_gs=False,
            ref_view_strategy="middle",
        )
    except Exception as e:
        del model
        torch.cuda.empty_cache()
        raise RuntimeError(f"DA3 inference failed: {e}")

    depths = prediction.depth               # (N, H, W)
    extrinsics = prediction.extrinsics      # (N, 3, 4) w2c
    intrinsics = prediction.intrinsics      # (N, 3, 3)
    proc_imgs = prediction.processed_images # (N, H, W, 3) uint8

    print(f"DA3 done. Depth: {depths.shape}, Extrinsics: {extrinsics.shape}")

    del model
    torch.cuda.empty_cache()

    N, H, W = depths.shape
    pixels_per_frame = max(1, max_points // N)
    step = max(1, int(np.sqrt(H * W / pixels_per_frame)))
    print(f"Sampling every {step} pixels/frame ({pixels_per_frame} target/frame)")

    v_idx, u_idx = np.mgrid[0:H:step, 0:W:step]
    v_flat = v_idx.ravel().astype(np.float32)
    u_flat = u_idx.ravel().astype(np.float32)

    all_points = []
    all_colors = []

    for i in range(N):
        depth = depths[i]
        K = intrinsics[i]       # (3, 3)
        E = extrinsics[i]       # (3, 4) w2c

        d = depth[v_idx, u_idx].ravel()
        colors_rgb = proc_imgs[i][v_idx, u_idx].reshape(-1, 3)

        valid = d > 1e-3
        d = d[valid]
        u_v = u_flat[valid]
        v_v = v_flat[valid]
        colors_v = colors_rgb[valid].astype(np.float32) / 255.0
        if len(d) == 0:
            continue

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x_cam = (u_v - cx) * d / fx
        y_cam = (v_v - cy) * d / fy
        z_cam = d
        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

        # Extrinsics are (3, 4) w2c — extract R (3,3) and t (3,)
        R = E[:3, :3]
        t = E[:3, 3]
        R_inv = R.T
        t_inv = -R.T @ t
        pts_world = (R_inv @ pts_cam.T).T + t_inv

        all_points.append(pts_world.astype(np.float32))
        all_colors.append(colors_v)

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx]

    print(f"Point cloud: {len(points)} points")

    ply_path = os.path.join(export_dir, "reconstruction.ply")
    _save_pointcloud_ply(points, colors, ply_path)
    return ply_path


def _save_pointcloud_ply(points: np.ndarray, colors: np.ndarray, path: str):
    """Save colored point cloud as binary PLY."""
    N = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    # Build binary data with numpy for speed
    rgb = np.clip(colors * 255, 0, 255).astype(np.uint8)
    # Interleave: 3 floats + 3 bytes = 15 bytes per point
    dt = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
    ])
    structured = np.empty(N, dtype=dt)
    structured['x'] = points[:, 0]
    structured['y'] = points[:, 1]
    structured['z'] = points[:, 2]
    structured['r'] = rgb[:, 0]
    structured['g'] = rgb[:, 1]
    structured['b'] = rgb[:, 2]

    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(structured.tobytes())

    print(f"Saved PLY: {path} ({os.path.getsize(path) / 1024 / 1024:.1f} MB)")


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
    """RunPod serverless entrypoint."""
    job_input = event["input"]
    job_id = event.get("id", "unknown")

    # Diagnostic mode — test imports, GPU, model loading
    if job_input.get("diagnose"):
        diag = {"gpu": False, "imports": [], "errors": []}
        try:
            import torch
            diag["torch"] = torch.__version__
            diag["gpu"] = torch.cuda.is_available()
            diag["gpu_name"] = torch.cuda.get_device_name(0) if diag["gpu"] else "N/A"
            diag["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1) if diag["gpu"] else 0
        except Exception as e:
            diag["errors"].append(f"torch: {e}")
        try:
            from depth_anything_3.api import DepthAnything3
            diag["imports"].append("depth_anything_3.api")
        except Exception as e:
            diag["errors"].append(f"da3 import: {e}")
        try:
            from plyfile import PlyData
            diag["imports"].append("plyfile")
        except Exception as e:
            diag["errors"].append(f"plyfile: {e}")
        try:
            from PIL import Image
            diag["imports"].append("PIL")
        except Exception as e:
            diag["errors"].append(f"PIL: {e}")
        try:
            from addict import Dict as AddictDict
            diag["imports"].append("addict")
        except Exception as e:
            diag["errors"].append(f"addict: {e}")
        # Test model loading if requested
        if job_input.get("load_model"):
            try:
                import torch
                from depth_anything_3.api import DepthAnything3
                model = DepthAnything3.from_pretrained(MODEL_ID)
                diag["model_loaded"] = True
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                diag["errors"].append(f"model load: {e}")
                diag["model_loaded"] = False
        return diag

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

        # Step 3: DA3 — depth estimation + camera poses → 3D point cloud
        runpod.serverless.progress_update(event, {"progress": 20, "message": "Running Depth Anything V3 reconstruction..."})
        ply_path = run_da3_reconstruction(images_dir, export_dir)

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
    print(f"Pre-downloading Depth Anything V3 model weights ({MODEL_ID})...")
    try:
        from depth_anything_3.api import DepthAnything3
        _model = DepthAnything3.from_pretrained(MODEL_ID)
        del _model
        import torch
        torch.cuda.empty_cache()
        print("Model weights cached successfully.")
    except Exception as e:
        print(f"WARNING: Could not pre-download model: {e}")
        import traceback
        traceback.print_exc()
        print("Model will be downloaded on first job.")

    runpod.serverless.start({"handler": handler})
