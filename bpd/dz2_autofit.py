#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""
dz2_autofit.py — Automatic parameter fitting of d_z² orbital to UAP IR frames.

Uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) to find
the orbital orientation, shape, and brightness parameters that best match
the observed IR silhouette.

Pipeline:
  1. Extract target silhouette from IR frame (adaptive threshold)
  2. Render d_z² at candidate parameters → green-screen → silhouette
  3. Compare via Chamfer distance + IoU
  4. CMA-ES minimizes the distance
"""
import numpy as np
from scipy.special import sph_harm
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
from scipy.spatial.distance import cdist
from PIL import Image
import time, sys, os

# ═══════════════════════════════════════════════════════════════
# STEP 1: Extract target silhouette from IR frame
# ═══════════════════════════════════════════════════════════════

def extract_target_silhouette(frame_path, crop_center=None, crop_size=200):
    """
    Extract object silhouette from IR frame.
    Returns binary mask + edge points + cropped grayscale.
    """
    img = np.array(Image.open(frame_path).convert('L')).astype(np.float64)
    h, w = img.shape
    
    # Auto-detect center if not provided
    if crop_center is None:
        # Find the region with highest contrast (the object)
        # Compute local std in sliding window
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(img, size=50)
        local_sq_mean = uniform_filter(img**2, size=50)
        local_var = local_sq_mean - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 0))
        
        # Find peak of local std (object location)
        # Mask out the HUD elements (edges of frame)
        margin = 100
        local_std[:margin, :] = 0; local_std[-margin:, :] = 0
        local_std[:, :margin] = 0; local_std[:, -margin:] = 0
        
        cy, cx = np.unravel_index(np.argmax(local_std), local_std.shape)
        crop_center = (cx, cy)
    
    cx, cy = crop_center
    
    # Crop around object
    x0 = max(0, cx - crop_size); x1 = min(w, cx + crop_size)
    y0 = max(0, cy - crop_size); y1 = min(h, cy + crop_size)
    crop = img[y0:y1, x0:x1]
    
    # Normalize
    crop_norm = (crop - crop.min()) / (crop.max() - crop.min() + 1e-10)
    
    # Adaptive threshold: object pixels deviate from background
    bg_mean = np.median(crop_norm)
    bg_std = np.std(crop_norm[crop_norm < np.percentile(crop_norm, 75)])
    
    # Object = pixels significantly brighter OR darker than background
    bright_mask = crop_norm > (bg_mean + 2.0 * bg_std)
    dark_mask = crop_norm < (bg_mean - 2.5 * bg_std)
    object_mask = bright_mask | dark_mask
    
    # Clean up: remove small noise, fill holes
    object_mask = binary_dilation(object_mask, iterations=2)
    object_mask = binary_erosion(object_mask, iterations=1)
    
    # Extract edge points for Chamfer distance
    from scipy.ndimage import sobel
    edges_y = sobel(object_mask.astype(float), axis=0)
    edges_x = sobel(object_mask.astype(float), axis=1)
    edge_mag = np.sqrt(edges_x**2 + edges_y**2)
    edge_points = np.argwhere(edge_mag > 0.5)  # [N, 2] array of (row, col)
    
    return {
        'crop': crop_norm,
        'mask': object_mask,
        'edge_points': edge_points,
        'crop_center': crop_center,
        'crop_size': crop_size,
        'crop_bounds': (x0, y0, x1, y1),
    }


# ═══════════════════════════════════════════════════════════════
# STEP 2: Fast renderer (optimized for optimizer — no GUI, no extras)
# ═══════════════════════════════════════════════════════════════

def rotation_matrix_from_zaxis(z_target):
    z_target = z_target / np.linalg.norm(z_target)
    z_orig = np.array([0.0, 0.0, 1.0])
    v = np.cross(z_orig, z_target)
    s = np.linalg.norm(v)
    c = np.dot(z_orig, z_target)
    if s < 1e-10:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)


def render_fast(params, img_size=200):
    """
    Fast render for optimizer. Returns silhouette mask + brightness image.
    params = [zx, zy, zz, pleat, z_scale, z_rot, threshold]
    """
    zx, zy, zz, pleat, z_scale, z_rot, threshold = params
    
    n_theta, n_phi = 100, 200  # lower res for speed
    
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta, phi)
    
    # Y₂₀
    Y20 = np.real(sph_harm(0, 2, PHI, THETA))
    R = np.abs(Y20)
    
    # Pleating
    if abs(pleat) > 0.001:
        equatorial = np.sin(THETA) ** 2
        pleat_mod = np.cos(3 * PHI) * equatorial
        pleat_bias = equatorial * 0.3
        R = R + pleat * (pleat_mod + pleat_bias)
        R = np.maximum(R, 0)
    
    # Cartesian + z-scale
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA) * z_scale
    
    # Z-rotation
    if abs(z_rot) > 0.001:
        c_r, s_r = np.cos(z_rot), np.sin(z_rot)
        X, Y = X * c_r - Y * s_r, X * s_r + Y * c_r
    
    # Z+ only
    mask_surf = (R > threshold * np.max(R)) & (Z >= -0.08 * z_scale)
    
    # Rotate
    z_axis = np.array([zx, zy, zz])
    nl = np.linalg.norm(z_axis)
    if nl < 1e-10:
        return np.zeros((img_size, img_size)), np.zeros((img_size, img_size), dtype=bool)
    z_axis = z_axis / nl
    Rot = rotation_matrix_from_zaxis(z_axis)
    
    pts = Rot @ np.stack([X[mask_surf].flatten(), Y[mask_surf].flatten(), Z[mask_surf].flatten()])
    rx, ry, rz = pts[0], pts[1], pts[2]
    
    if len(rx) == 0:
        return np.zeros((img_size, img_size)), np.zeros((img_size, img_size), dtype=bool)
    
    margin = 1.2
    max_ext = max(np.max(np.abs(rx)), np.max(np.abs(ry)))
    scale = img_size / (2 * margin * max(max_ext, 0.01))
    
    ix = (rx * scale + img_size / 2).astype(int)
    iy = (img_size / 2 - ry * scale).astype(int)
    
    img = np.zeros((img_size, img_size), dtype=np.float64)
    depth = np.full((img_size, img_size), -np.inf)
    valid = (ix >= 0) & (ix < img_size) & (iy >= 0) & (iy < img_size)
    
    # Emission model brightness
    nrm = Rot @ np.stack([
        X[mask_surf].flatten(), Y[mask_surf].flatten(), Z[mask_surf].flatten()])
    rnz = nrm[2]  # normal z-component (facing camera)
    
    for i in range(len(ix)):
        if valid[i] and rz[i] > depth[iy[i], ix[i]]:
            depth[iy[i], ix[i]] = rz[i]
            img[iy[i], ix[i]] = 0.15 + 0.85 * abs(rnz[i])
    
    img = gaussian_filter(img, sigma=1.0)
    mask = img > 0.05
    
    return img, mask


# ═══════════════════════════════════════════════════════════════
# STEP 3: Objective function (what CMA-ES minimizes)
# ═══════════════════════════════════════════════════════════════

def chamfer_distance(points_a, points_b):
    """Average nearest-neighbor distance between two point sets."""
    if len(points_a) == 0 or len(points_b) == 0:
        return 1000.0
    # Subsample for speed
    max_pts = 500
    if len(points_a) > max_pts:
        idx = np.random.choice(len(points_a), max_pts, replace=False)
        points_a = points_a[idx]
    if len(points_b) > max_pts:
        idx = np.random.choice(len(points_b), max_pts, replace=False)
        points_b = points_b[idx]
    
    dists = cdist(points_a, points_b)
    d_ab = np.mean(np.min(dists, axis=1))
    d_ba = np.mean(np.min(dists, axis=0))
    return (d_ab + d_ba) / 2.0


def iou(mask_a, mask_b):
    """Intersection over Union of two binary masks."""
    intersection = np.sum(mask_a & mask_b)
    union = np.sum(mask_a | mask_b)
    if union == 0:
        return 0.0
    return intersection / union


def objective(params, target_info, img_size=200):
    """
    Compute fit score: lower = better match.
    Combines Chamfer distance + (1 - IoU) + brightness correlation.
    """
    try:
        rendered_img, rendered_mask = render_fast(params, img_size=img_size)
    except:
        return 1000.0
    
    if np.sum(rendered_mask) < 10:
        return 1000.0  # degenerate render
    
    target_mask = target_info['mask']
    target_edges = target_info['edge_points']
    
    # Resize target to match render size
    from PIL import Image as PILImage
    target_mask_resized = np.array(PILImage.fromarray(target_mask.astype(np.uint8) * 255).resize(
        (img_size, img_size), PILImage.NEAREST)) > 127
    
    target_edges_resized = target_info['edge_points'].copy().astype(float)
    target_h, target_w = target_info['mask'].shape
    target_edges_resized[:, 0] *= img_size / target_h
    target_edges_resized[:, 1] *= img_size / target_w
    
    # Rendered edge points
    from scipy.ndimage import sobel
    re_y = sobel(rendered_mask.astype(float), axis=0)
    re_x = sobel(rendered_mask.astype(float), axis=1)
    re_mag = np.sqrt(re_y**2 + re_x**2)
    rendered_edges = np.argwhere(re_mag > 0.5)
    
    # Chamfer distance (edge alignment)
    chamfer = chamfer_distance(rendered_edges, target_edges_resized)
    
    # IoU (area overlap)
    iou_score = iou(rendered_mask, target_mask_resized)
    
    # Brightness correlation (does the rendered brightness pattern match?)
    target_crop = target_info['crop']
    target_crop_resized = np.array(PILImage.fromarray(
        (target_crop * 255).astype(np.uint8)).resize((img_size, img_size))) / 255.0
    
    # Only compare in the union of both masks
    union_mask = rendered_mask | target_mask_resized
    if np.sum(union_mask) > 10:
        r_vals = rendered_img[union_mask]
        t_vals = target_crop_resized[union_mask]
        if np.std(r_vals) > 0.01 and np.std(t_vals) > 0.01:
            corr = np.corrcoef(r_vals, t_vals)[0, 1]
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0
    else:
        corr = 0.0
    
    # Combined score: Chamfer + (1-IoU) + (1-corr)
    score = chamfer / 10.0 + 2.0 * (1.0 - iou_score) + 0.5 * (1.0 - corr)
    
    return score


# ═══════════════════════════════════════════════════════════════
# STEP 4: CMA-ES optimizer
# ═══════════════════════════════════════════════════════════════

def optimize_fit(target_info, n_iterations=200, population_size=20, img_size=200):
    """
    Run CMA-ES to find optimal d_z² parameters matching the target.
    """
    # Initial guess (from our manual fit)
    x0 = np.array([0.77, 0.21, 0.60, 0.16, 2.1, 0.0, 0.15])
    
    # Initial step sizes
    sigma0 = 0.3
    
    # Bounds
    bounds_lo = np.array([-1.0, -1.0, -1.0, 0.0, 0.5, -3.14, 0.05])
    bounds_hi = np.array([ 1.0,  1.0,  1.0, 0.5, 4.0,  3.14, 0.50])
    
    # Simple CMA-ES implementation (no external dependency)
    dim = len(x0)
    mean = x0.copy()
    C = np.eye(dim)  # covariance matrix
    sigma = sigma0
    
    best_score = float('inf')
    best_params = x0.copy()
    
    # CMA-ES parameters
    lam = population_size
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mu_eff = 1.0 / np.sum(weights ** 2)
    
    cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
    cs = (mu_eff + 2) / (dim + mu_eff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
    cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
    damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
    
    pc = np.zeros(dim)
    ps = np.zeros(dim)
    
    chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
    
    eval_count = 0
    t_start = time.time()
    
    print(f"CMA-ES optimization: {dim}D, pop={lam}, max_iter={n_iterations}")
    print(f"  Initial params: zx={x0[0]:.2f} zy={x0[1]:.2f} zz={x0[2]:.2f} "
          f"pleat={x0[3]:.2f} zscale={x0[4]:.1f} zrot={x0[5]:.2f} thresh={x0[6]:.2f}")
    
    for gen in range(n_iterations):
        # Sample population
        try:
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            sqrt_C = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        except:
            sqrt_C = np.eye(dim)
        
        solutions = []
        for k in range(lam):
            z = np.random.randn(dim)
            x = mean + sigma * (sqrt_C @ z)
            # Clip to bounds
            x = np.clip(x, bounds_lo, bounds_hi)
            score = objective(x, target_info, img_size=img_size)
            solutions.append((score, x, z))
            eval_count += 1
        
        # Sort by fitness
        solutions.sort(key=lambda s: s[0])
        
        if solutions[0][0] < best_score:
            best_score = solutions[0][0]
            best_params = solutions[0][1].copy()
        
        # Update mean
        old_mean = mean.copy()
        mean = np.zeros(dim)
        for i in range(mu):
            mean += weights[i] * solutions[i][1]
        
        # Update evolution paths
        invsqrt_C = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (invsqrt_C @ (mean - old_mean) / sigma)
        
        hs = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chiN else 0
        pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mu_eff) * (mean - old_mean) / sigma
        
        # Update covariance
        C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hs) * cc * (2 - cc) * C)
        for i in range(mu):
            diff = (solutions[i][1] - old_mean) / sigma
            C += cmu * weights[i] * np.outer(diff, diff)
        
        # Update sigma
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = max(sigma, 1e-10)
        
        # Report progress
        if gen % 20 == 0 or gen == n_iterations - 1:
            elapsed = time.time() - t_start
            p = best_params
            print(f"  Gen {gen:4d}: best={best_score:.4f} "
                  f"z=[{p[0]:.2f},{p[1]:.2f},{p[2]:.2f}] "
                  f"pleat={p[3]:.2f} zscale={p[4]:.1f} "
                  f"zrot={p[5]:.2f} thresh={p[6]:.2f} "
                  f"sigma={sigma:.3f} ({elapsed:.1f}s)")
    
    elapsed = time.time() - t_start
    print(f"\nOptimization complete: {eval_count} evaluations in {elapsed:.1f}s")
    print(f"Best score: {best_score:.4f}")
    print(f"Best params:")
    names = ['zx', 'zy', 'zz', 'pleat', 'z_scale', 'z_rot', 'threshold']
    for name, val in zip(names, best_params):
        print(f"  {name:12s} = {val:.4f}")
    
    return best_params, best_score


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    frame_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/pr46_frames/frame_0050.png'
    n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    print(f"=== d_z² Autofit — CMA-ES Parameter Optimizer ===")
    print(f"Target frame: {frame_path}")
    
    # Step 1: Extract target
    print("\n--- Extracting target silhouette ---")
    target = extract_target_silhouette(frame_path, crop_size=150)
    print(f"  Center: {target['crop_center']}")
    print(f"  Object pixels: {np.sum(target['mask'])}")
    print(f"  Edge points: {len(target['edge_points'])}")
    
    # Save target visualization
    target_vis = (target['crop'] * 255).astype(np.uint8)
    Image.fromarray(target_vis).save('/tmp/dz2_autofit_target.png')
    mask_vis = (target['mask'].astype(np.uint8) * 255)
    Image.fromarray(mask_vis).save('/tmp/dz2_autofit_target_mask.png')
    print("  Saved target + mask visualizations")
    
    # Step 2: Optimize
    print("\n--- Running CMA-ES optimization ---")
    best_params, best_score = optimize_fit(target, n_iterations=n_iter)
    
    # Step 3: Render best fit at high resolution
    print("\n--- Rendering best fit ---")
    from dz2_render_v2 import render_dz2, save_render
    
    rgb, mask, z_axis = render_dz2(
        zx=best_params[0], zy=best_params[1], zz=best_params[2],
        pleat=best_params[3], z_scale=best_params[4], z_rot=best_params[5],
        threshold=best_params[6], img_size=800, brightness_model='emission',
    )
    save_render(rgb, mask, z_axis, {}, '/tmp/dz2_autofit_best.png')
    
    print("\nDone! Files saved:")
    print("  /tmp/dz2_autofit_target.png      — cropped IR frame")
    print("  /tmp/dz2_autofit_target_mask.png  — extracted silhouette")
    print("  /tmp/dz2_autofit_best.png         — best-fit render (green screen)")
