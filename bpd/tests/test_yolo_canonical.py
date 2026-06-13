#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
# Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
"""YOLO Canonical Test Harness — Stock vs BPD Comparison

Downloads canonical COCO val2017 test images, runs stock YOLOv5
(via torch.hub), saves reference detections. Then compares against
our BPD-generated YOLO inference.

Phase 1: Generate canonical reference outputs from stock YOLOv5
Phase 2: Compare BPD YOLO outputs against reference
Phase 3: Performance measurement (if outputs match)

Author: medayek (Collective SME, Verification Methodology)
Date: 2026-05-21
Per Heath: identical output, then measure performance.
"""

import torch
import numpy as np
import os
import time
import json
import urllib.request
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path('/tmp/yolo_canonical')
OUTPUT_DIR.mkdir(exist_ok=True)

# 10 diverse COCO val2017 images (public URLs)
# Selected for diversity: people, vehicles, animals, indoor, outdoor
COCO_TEST_IMAGES = {
    '000000397133': 'http://images.cocodataset.org/val2017/000000397133.jpg',  # street scene
    '000000037777': 'http://images.cocodataset.org/val2017/000000037777.jpg',  # baseball
    '000000252219': 'http://images.cocodataset.org/val2017/000000252219.jpg',  # cat
    '000000087038': 'http://images.cocodataset.org/val2017/000000087038.jpg',  # food/dining
    '000000174482': 'http://images.cocodataset.org/val2017/000000174482.jpg',  # surfing
    '000000403385': 'http://images.cocodataset.org/val2017/000000403385.jpg',  # bus
    '000000006818': 'http://images.cocodataset.org/val2017/000000006818.jpg',  # kitchen
    '000000480985': 'http://images.cocodataset.org/val2017/000000480985.jpg',  # skiing
    '000000458054': 'http://images.cocodataset.org/val2017/000000458054.jpg',  # dog
    '000000331352': 'http://images.cocodataset.org/val2017/000000331352.jpg',  # traffic
}

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Download images + generate stock YOLOv5 reference
# ═══════════════════════════════════════════════════════════════════════

def download_test_images():
    """Download COCO val2017 test images."""
    img_dir = OUTPUT_DIR / 'images'
    img_dir.mkdir(exist_ok=True)
    
    downloaded = []
    for img_id, url in COCO_TEST_IMAGES.items():
        img_path = img_dir / f'{img_id}.jpg'
        if not img_path.exists():
            print(f"  Downloading {img_id}...")
            urllib.request.urlretrieve(url, img_path)
        downloaded.append(str(img_path))
    
    print(f"  {len(downloaded)} test images ready")
    return downloaded


def generate_stock_reference(image_paths):
    """Run stock YOLOv5 on test images and save canonical detections."""
    print("\n=== Loading stock YOLOv5n via torch.hub ===")
    
    # Load YOLOv5n from Ultralytics via torch.hub
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    model.eval()
    model.conf = CONFIDENCE_THRESHOLD
    model.iou = IOU_THRESHOLD
    
    ref_dir = OUTPUT_DIR / 'reference'
    ref_dir.mkdir(exist_ok=True)
    
    all_results = {}
    total_time = 0
    
    for img_path in image_paths:
        img_id = Path(img_path).stem
        print(f"\n  Processing {img_id}...")
        
        # Inference with timing
        t0 = time.perf_counter()
        results = model(img_path)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        total_time += elapsed_ms
        
        # Extract detections
        det = results.pandas().xyxy[0]
        
        detections = {
            'boxes': results.xyxy[0][:, :4].cpu().numpy().astype(np.float32),
            'confidence': results.xyxy[0][:, 4].cpu().numpy().astype(np.float32),
            'class_ids': results.xyxy[0][:, 5].cpu().numpy().astype(np.int32),
            'class_names': det['name'].tolist() if len(det) > 0 else [],
            'n_detections': len(det),
            'inference_ms': elapsed_ms,
        }
        
        # Save as .npz
        np.savez(
            ref_dir / f'{img_id}_ref.npz',
            boxes=detections['boxes'],
            confidence=detections['confidence'],
            class_ids=detections['class_ids'],
        )
        
        # Save human-readable summary
        all_results[img_id] = {
            'n_detections': detections['n_detections'],
            'classes': detections['class_names'],
            'inference_ms': round(elapsed_ms, 1),
        }
        
        print(f"    {detections['n_detections']} detections, {elapsed_ms:.1f}ms")
        for i in range(min(5, len(det))):
            row = det.iloc[i]
            print(f"      {row['name']:15s} conf={row['confidence']:.3f} "
                  f"box=[{row['xmin']:.0f},{row['ymin']:.0f},{row['xmax']:.0f},{row['ymax']:.0f}]")
    
    # Save summary
    avg_ms = total_time / len(image_paths)
    summary = {
        'model': 'yolov5n',
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'iou_threshold': IOU_THRESHOLD,
        'n_images': len(image_paths),
        'avg_inference_ms': round(avg_ms, 1),
        'total_inference_ms': round(total_time, 1),
        'per_image': all_results,
    }
    
    with open(OUTPUT_DIR / 'reference_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  Average inference: {avg_ms:.1f}ms/image (stock YOLOv5n)")
    print(f"  Reference saved to {ref_dir}")
    return summary


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Compare BPD YOLO output against reference
# ═══════════════════════════════════════════════════════════════════════

def compare_detections(ref_path, our_path, tolerance_pixels=1.0):
    """Compare two detection sets for equivalence.
    
    Returns (match, details) where match is True if detections are
    functionally equivalent (same classes, similar boxes, similar confidence).
    """
    ref = np.load(ref_path)
    our = np.load(our_path)
    
    ref_boxes = ref['boxes']
    ref_conf = ref['confidence']
    ref_cls = ref['class_ids']
    
    our_boxes = our['boxes']
    our_conf = our['confidence']
    our_cls = our['class_ids']
    
    # Check: same number of detections
    if len(ref_boxes) != len(our_boxes):
        return False, f"Detection count mismatch: ref={len(ref_boxes)} ours={len(our_boxes)}"
    
    if len(ref_boxes) == 0:
        return True, "Both empty"
    
    # Check: same class labels (sorted by confidence)
    ref_order = np.argsort(-ref_conf)
    our_order = np.argsort(-our_conf)
    
    if not np.array_equal(ref_cls[ref_order], our_cls[our_order]):
        return False, f"Class mismatch: ref={ref_cls[ref_order].tolist()} ours={our_cls[our_order].tolist()}"
    
    # Check: confidence bit-identical
    conf_bits_match = np.array_equal(
        ref_conf[ref_order].view(np.uint32),
        our_conf[our_order].view(np.uint32)
    )
    
    # Check: boxes within tolerance
    box_diff = np.abs(ref_boxes[ref_order] - our_boxes[our_order])
    max_box_diff = box_diff.max() if len(box_diff) > 0 else 0
    
    # Confidence ULP
    if not conf_bits_match:
        ref_bits = ref_conf[ref_order].view(np.uint32)
        our_bits = our_conf[our_order].view(np.uint32)
        max_conf_ulp = int(np.max(np.abs(ref_bits.astype(np.int64) - our_bits.astype(np.int64))))
    else:
        max_conf_ulp = 0
    
    details = {
        'n_detections': len(ref_boxes),
        'classes_match': True,
        'conf_bit_identical': conf_bits_match,
        'max_conf_ulp': max_conf_ulp,
        'max_box_diff_pixels': float(max_box_diff),
        'boxes_within_tolerance': max_box_diff <= tolerance_pixels,
    }
    
    is_match = (details['classes_match'] and 
                details['conf_bit_identical'] and
                details['boxes_within_tolerance'])
    
    return is_match, details


def run_comparison():
    """Compare BPD YOLO output against stock reference."""
    ref_dir = OUTPUT_DIR / 'reference'
    bpd_dir = OUTPUT_DIR / 'bpd_output'
    
    if not bpd_dir.exists():
        print("\n  BPD output directory not found. Run BPD YOLO first.")
        print(f"  Expected: {bpd_dir}/<image_id>_bpd.npz")
        return
    
    print("\n=== Comparing BPD YOLO vs Stock Reference ===")
    
    results = {}
    for img_id in COCO_TEST_IMAGES:
        ref_path = ref_dir / f'{img_id}_ref.npz'
        bpd_path = bpd_dir / f'{img_id}_bpd.npz'
        
        if not ref_path.exists():
            print(f"  SKIP {img_id}: no reference")
            continue
        if not bpd_path.exists():
            print(f"  SKIP {img_id}: no BPD output")
            continue
        
        match, details = compare_detections(ref_path, bpd_path)
        results[img_id] = {'match': match, 'details': details}
        
        symbol = '✅' if match else '❌'
        if isinstance(details, str):
            print(f"  {symbol} {img_id}: {details}")
        else:
            print(f"  {symbol} {img_id}: {details['n_detections']} detections, "
                  f"conf_ULP={details['max_conf_ulp']}, "
                  f"box_diff={details['max_box_diff_pixels']:.3f}px")
    
    # Summary
    n_match = sum(1 for v in results.values() if v['match'])
    n_total = len(results)
    print(f"\n  MATCH: {n_match}/{n_total}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Performance measurement
# ═══════════════════════════════════════════════════════════════════════

def measure_performance(image_paths, n_warmup=3, n_runs=10):
    """Measure inference performance on stock YOLOv5 for baseline."""
    print(f"\n=== Performance Measurement (stock YOLOv5n) ===")
    print(f"  Warmup: {n_warmup} runs, Measurement: {n_runs} runs")
    
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    model.eval()
    model.conf = CONFIDENCE_THRESHOLD
    
    # Warmup
    for _ in range(n_warmup):
        for img in image_paths[:2]:
            _ = model(img)
    
    # Measure
    times = []
    for run in range(n_runs):
        t0 = time.perf_counter()
        for img in image_paths:
            _ = model(img)
        t1 = time.perf_counter()
        per_image_ms = (t1 - t0) / len(image_paths) * 1000
        times.append(per_image_ms)
    
    times = np.array(times)
    print(f"  Mean:   {times.mean():.1f} ms/image")
    print(f"  Std:    {times.std():.1f} ms")
    print(f"  Min:    {times.min():.1f} ms")
    print(f"  Max:    {times.max():.1f} ms")
    print(f"  Median: {np.median(times):.1f} ms")
    
    return {
        'mean_ms': round(float(times.mean()), 1),
        'std_ms': round(float(times.std()), 1),
        'min_ms': round(float(times.min()), 1),
        'max_ms': round(float(times.max()), 1),
        'n_images': len(image_paths),
        'n_runs': n_runs,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    
    print("=" * 70)
    print("YOLO Canonical Test Harness")
    print("=" * 70)
    
    # Phase 1: Download + generate reference
    print("\n--- Phase 1: Download test images + generate stock reference ---")
    image_paths = download_test_images()
    
    if '--reference' in sys.argv or '--all' in sys.argv:
        summary = generate_stock_reference(image_paths)
    
    # Phase 2: Compare
    if '--compare' in sys.argv or '--all' in sys.argv:
        run_comparison()
    
    # Phase 3: Performance
    if '--perf' in sys.argv or '--all' in sys.argv:
        perf = measure_performance(image_paths)
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  --reference  Generate stock YOLOv5 canonical outputs")
        print("  --compare    Compare BPD YOLO vs stock reference")
        print("  --perf       Measure stock YOLOv5 performance baseline")
        print("  --all        Run all phases")
