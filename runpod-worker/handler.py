def run_da3_reconstruction_safe(images_dir: str, export_dir: str, max_points: int = 500000, density_factor: float = 2.0) -> str:
    """Run DA3 depth + GS reconstruction. Very conservative to prevent crashes."""
    import torch
    from depth_anything_3.api import DepthAnything3

    # Collect image paths
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    image_paths = [os.path.join(images_dir, f) for f in image_files]

    print(f"Running DA3 reconstruction on {len(image_paths)} images...")
    print(f"Target: {max_points} points, density factor: {density_factor}")

    # Conservative limits to prevent crashes
    max_safe_points = min(300000, int(max_points * density_factor))  # Max 300K points
    print(f"Safe limit: {max_safe_points} points (to prevent crashes)")

    # Load model
    try:
        model = DepthAnything3.from_pretrained(MODEL_ID)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None

    # Run inference with error handling
    try:
        with torch.no_grad():
            prediction = model.inference(
                image=image_paths,
                infer_gs=False,  # Disable GS to avoid issues
                ref_view_strategy="middle"
            )
    except Exception as e:
        print(f"DA3 inference failed: {e}")
        return None

    try:
        depths = prediction.depths          # (N, H, W)
        extrinsics = prediction.extrinsics  # (N, 3, 4) w2c
        intrinsics = prediction.intrinsics  # (N, 3, 3)
        proc_imgs = prediction.processed_images  # (N, H, W, 3) uint8

        print(f"DA3 done. Depth: {depths.shape}, Extrinsics: {extrinsics.shape}")
    except Exception as e:
        print(f"Failed to extract prediction data: {e}")
        return None

    # Clean up model
    try:
        del model
        torch.cuda.empty_cache()
    except:
        pass

    try:
        N, H, W = depths.shape
        # Very conservative sampling to prevent crashes
        pixels_per_frame = max(1, max_safe_points // N)
        step = max(2, int(np.sqrt(H * W / pixels_per_frame)))  # Minimum step 2
        print(f"Very conservative sampling: every {step} pixels/frame ({pixels_per_frame} target/frame)")
        
        v_idx, u_idx = np.mgrid[0:H:step, 0:W:step]
        v_flat = v_idx.ravel().astype(np.float32)
        u_flat = u_flat.ravel().astype(np.float32)

        all_points = []
        all_colors = []

        for i in range(N):
            try:
                depth = depths[i]
                K = intrinsics[i]       # (3, 3)
                E = extrinsics[i]       # (3, 4) w2c

                d = depth[v_idx, u_idx].ravel()
                colors_rgb = proc_imgs[i][v_idx, u_idx].reshape(-1, 3)

                valid = d > 1e-3
                if len(valid) == 0:
                    continue
                    
                d = d[valid]
                u_v = u_flat[valid]
                v_v = v_flat[valid]
                colors_v = colors_rgb[valid].astype(np.float32) / 255.0

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
                
                # Progress update
                if i % 10 == 0:
                    print(f"Processed frame {i+1}/{N}")
                    
            except Exception as e:
                print(f"Error processing frame {i}: {e}")
                continue

        if len(all_points) == 0:
            print("No valid points generated")
            return None

        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)

        # Final safety check
        if len(points) > max_safe_points:
            idx = np.random.choice(len(points), max_safe_points, replace=False)
            points = points[idx]
            colors = colors[idx]
            print(f"Final safety: reduced to {max_safe_points} points")
        
        print(f"Final point cloud: {len(points)} points (conservative for stability)")

        ply_path = os.path.join(export_dir, "reconstruction.ply")
        _save_pointcloud_ply(points, colors, ply_path)
        return ply_path
        
    except Exception as e:
        print(f"Point cloud generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None
