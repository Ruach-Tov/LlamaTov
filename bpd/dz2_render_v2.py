#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""
dz2_render_v2.py — d_z² orbital renderer for IR frame comparison

OUTPUT: Green-screen background (chroma key), pure white-black object.
No color tones. Pure monochrome luminance for silhouette fit-testing.

Two brightness models:
  A) EMISSION: brightness ∝ N·V (surface normal facing camera)
  B) REFLECTION: brightness ∝ N·L (surface normal facing light source)

Usage (headless, no GUI needed):
    python3 dz2_render_v2.py [--model emission|reflection] [--sun-az 45] [--sun-el 30]
"""
import numpy as np
from scipy.special import sph_harm
from scipy.ndimage import gaussian_filter
import argparse, os

# ═══════════════════════════════════════════════════════════════
# RENDERING ENGINE v2
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


def render_dz2(zx=0.77, zy=0.21, zz=0.60, pleat=0.16, threshold=0.15,
               z_scale=2.1, z_rot=0.0, img_size=800, n_theta=300, n_phi=600,
               brightness_model='emission', sun_az=45.0, sun_el=30.0,
               blur_sigma=2.0):
    """
    Render d_z² orbital as pure monochrome on green-screen background.
    
    Returns:
        rgb_image: [H, W, 3] uint8 — green background, white/black object
        mask: [H, W] bool — True where object pixels exist
    """
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta, phi)
    
    # Y₂₀ base shape
    Y20 = np.real(sph_harm(0, 2, PHI, THETA))
    R = np.abs(Y20)
    
    # Pleating: 3-fold torus modulation
    if abs(pleat) > 0.001:
        equatorial = np.sin(THETA) ** 2
        pleat_mod = np.cos(3 * PHI) * equatorial
        pleat_bias = equatorial * 0.3
        R = R + pleat * (pleat_mod + pleat_bias)
        R = np.maximum(R, 0)
    
    # Cartesian
    X = R * np.sin(THETA) * np.cos(PHI)
    Y = R * np.sin(THETA) * np.sin(PHI)
    Z = R * np.cos(THETA)
    
    # Z-scale
    Z = Z * z_scale
    
    # Z-rotation
    if abs(z_rot) > 0.001:
        c_r, s_r = np.cos(z_rot), np.sin(z_rot)
        X, Y = X * c_r - Y * s_r, X * s_r + Y * c_r
    
    # Compute surface normals BEFORE rotation (for lighting)
    # Approximate normals from the parametric surface
    # dS/dtheta and dS/dphi cross product
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    
    # Numerical gradient
    dXdt = np.gradient(X, dtheta, axis=1)
    dYdt = np.gradient(Y, dtheta, axis=1)
    dZdt = np.gradient(Z, dtheta, axis=1)
    dXdp = np.gradient(X, dphi, axis=0)
    dYdp = np.gradient(Y, dphi, axis=0)
    dZdp = np.gradient(Z, dphi, axis=0)
    
    # Normal = dS/dphi × dS/dtheta (outward pointing)
    NX = dYdp * dZdt - dZdp * dYdt
    NY = dZdp * dXdt - dXdp * dZdt
    NZ = dXdp * dYdt - dYdp * dXdt
    
    norm_len = np.sqrt(NX**2 + NY**2 + NZ**2)
    norm_len = np.maximum(norm_len, 1e-10)
    NX /= norm_len; NY /= norm_len; NZ /= norm_len
    
    # Z+ only mask
    mask_surf = (R > threshold * np.max(R)) & (Z >= -0.08 * z_scale)
    
    # View rotation
    z_axis = np.array([zx, zy, zz])
    nl = np.linalg.norm(z_axis)
    if nl < 1e-10:
        z_axis = np.array([0, 0, 1])
    else:
        z_axis = z_axis / nl
    Rot = rotation_matrix_from_zaxis(z_axis)
    
    # Rotate points
    x_f = X[mask_surf].flatten()
    y_f = Y[mask_surf].flatten()
    z_f = Z[mask_surf].flatten()
    nx_f = NX[mask_surf].flatten()
    ny_f = NY[mask_surf].flatten()
    nz_f = NZ[mask_surf].flatten()
    
    pts = Rot @ np.stack([x_f, y_f, z_f])
    nrm = Rot @ np.stack([nx_f, ny_f, nz_f])
    rx, ry, rz = pts[0], pts[1], pts[2]
    rnx, rny, rnz = nrm[0], nrm[1], nrm[2]
    
    # Projection
    margin = 1.2
    max_ext = max(np.max(np.abs(rx)) if len(rx) > 0 else 1,
                  np.max(np.abs(ry)) if len(ry) > 0 else 1)
    scale = img_size / (2 * margin * max(max_ext, 0.01))
    
    ix = (rx * scale + img_size / 2).astype(int)
    iy = (img_size / 2 - ry * scale).astype(int)
    
    # Depth buffer + brightness buffer
    img = np.zeros((img_size, img_size), dtype=np.float64)
    depth = np.full((img_size, img_size), -np.inf)
    valid = (ix >= 0) & (ix < img_size) & (iy >= 0) & (iy < img_size)
    
    # Brightness model
    if brightness_model == 'emission':
        # N·V where V = camera direction = [0, 0, 1] (looking along -z after rotation)
        # Actually camera looks along +z in screen space, so V = [0, 0, 1]
        brightness_vals = np.abs(rnz)  # facing camera = bright
    elif brightness_model == 'reflection':
        # N·L where L is the sun direction
        sun_az_rad = np.radians(sun_az)
        sun_el_rad = np.radians(sun_el)
        L = np.array([
            np.cos(sun_el_rad) * np.sin(sun_az_rad),
            np.sin(sun_el_rad),
            np.cos(sun_el_rad) * np.cos(sun_az_rad)
        ])
        L = L / np.linalg.norm(L)
        dot_nl = rnx * L[0] + rny * L[1] + rnz * L[2]
        brightness_vals = np.maximum(0, dot_nl)
    else:
        brightness_vals = np.ones(len(ix))  # flat white silhouette
    
    # Render with depth test
    for i in range(len(ix)):
        if valid[i] and rz[i] > depth[iy[i], ix[i]]:
            depth[iy[i], ix[i]] = rz[i]
            img[iy[i], ix[i]] = 0.15 + 0.85 * brightness_vals[i]
    
    # Gentle blur (atmospheric/optical)
    if blur_sigma > 0:
        img = gaussian_filter(img, sigma=blur_sigma)
    
    # Build RGB: green-screen background, grayscale object
    object_mask = img > 0.01
    
    rgb = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    # Green screen: pure green
    rgb[:, :, 0] = 0    # R
    rgb[:, :, 1] = 255   # G  
    rgb[:, :, 2] = 0    # B
    
    # Object: pure white-black monochrome (no tone)
    luminance = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    rgb[object_mask, 0] = luminance[object_mask]  # R
    rgb[object_mask, 1] = luminance[object_mask]  # G
    rgb[object_mask, 2] = luminance[object_mask]  # B
    
    return rgb, object_mask, z_axis


def save_render(rgb, mask, z_axis, params, filename):
    """Save the render as PNG."""
    from PIL import Image
    img = Image.fromarray(rgb)
    img.save(filename)
    print(f"Saved: {filename} ({rgb.shape[1]}x{rgb.shape[0]})")
    print(f"  z-axis: [{z_axis[0]:.3f}, {z_axis[1]:.3f}, {z_axis[2]:.3f}]")
    print(f"  Object pixels: {np.sum(mask)} / {mask.size} ({100*np.sum(mask)/mask.size:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="d_z² renderer v2 — IR comparison")
    parser.add_argument('--model', choices=['emission', 'reflection', 'silhouette'],
                        default='emission', help='Brightness model')
    parser.add_argument('--sun-az', type=float, default=45.0, help='Sun azimuth (degrees)')
    parser.add_argument('--sun-el', type=float, default=30.0, help='Sun elevation (degrees)')
    parser.add_argument('--zx', type=float, default=0.77)
    parser.add_argument('--zy', type=float, default=0.21)
    parser.add_argument('--zz', type=float, default=0.60)
    parser.add_argument('--pleat', type=float, default=0.16)
    parser.add_argument('--threshold', type=float, default=0.15)
    parser.add_argument('--z-scale', type=float, default=2.1)
    parser.add_argument('--z-rot', type=float, default=0.0)
    parser.add_argument('--size', type=int, default=800)
    parser.add_argument('--blur', type=float, default=2.0)
    parser.add_argument('--output', type=str, default='dz2_v2_greenscreen.png')
    args = parser.parse_args()
    
    print(f"Rendering d_z² orbital ({args.model} model)...")
    rgb, mask, z_axis = render_dz2(
        zx=args.zx, zy=args.zy, zz=args.zz,
        pleat=args.pleat, threshold=args.threshold,
        z_scale=args.z_scale, z_rot=args.z_rot,
        img_size=args.size, brightness_model=args.model,
        sun_az=args.sun_az, sun_el=args.sun_el,
        blur_sigma=args.blur,
    )
    
    save_render(rgb, mask, z_axis, vars(args), args.output)
    
    # Also render the other model for comparison
    other_model = 'reflection' if args.model == 'emission' else 'emission'
    print(f"\nAlso rendering {other_model} model...")
    rgb2, mask2, _ = render_dz2(
        zx=args.zx, zy=args.zy, zz=args.zz,
        pleat=args.pleat, threshold=args.threshold,
        z_scale=args.z_scale, z_rot=args.z_rot,
        img_size=args.size, brightness_model=other_model,
        sun_az=args.sun_az, sun_el=args.sun_el,
        blur_sigma=args.blur,
    )
    other_file = args.output.replace('.png', f'_{other_model}.png')
    save_render(rgb2, mask2, z_axis, vars(args), other_file)
