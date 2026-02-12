#!/usr/bin/env python3
"""
Compute Spearman correlation between predicted and ground-truth lesion volumes
for each model (baseline U-Net, concatenation, DAFT) and save results to CSV.

Usage:
    python code/spearman_volume_correlation.py

The script:
  1. For each model, loads all *_lab.nii.gz / *_pred.nii.gz pairs.
  2. Computes lesion volume (ml) for GT and prediction.
  3. Calculates Spearman rank correlation (rho, p-value).
  4. Saves per-subject volumes and the summary to CSV files.
"""

import os
import csv

import nibabel as nib
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration – paths to the three model result folders
# ---------------------------------------------------------------------------
BASE = '/media/cansu/DiskSpace/Cansu/ISLES24/ISLES24-Multimodal/data/challenge_test_set/test'

MODELS = {
    'Baseline U-Net': os.path.join(BASE, 'same_spacing_as_GT_Unet'),
    'Concatenation':  os.path.join(BASE, 'same_spacing_as_GT_Unet_CD'),
    'DAFT':           os.path.join(BASE, 'same_spacing_as_GT_Unet_DAFT'),
}

OUT_DIR = os.path.join(BASE, 'spearman_volume_results')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def volume_ml(mask_data: np.ndarray, voxel_vol_mm3: float) -> float:
    """Return the volume of a binary mask in millilitres."""
    return float(np.sum(mask_data.astype(bool)) * voxel_vol_mm3 / 1000.0)


def collect_volumes(results_path: str):
    """Return lists of (subject_id, gt_volume_ml, pred_volume_ml)."""
    lab_files = sorted(f for f in os.listdir(results_path) if f.endswith('_lab.nii.gz'))

    records = []
    for lab_name in tqdm(lab_files, desc=os.path.basename(results_path)):
        pred_name = lab_name.replace('_lab.nii.gz', '_pred.nii.gz')
        lab_path = os.path.join(results_path, lab_name)
        pred_path = os.path.join(results_path, pred_name)

        if not os.path.exists(pred_path):
            print(f'  Warning: prediction missing for {lab_name}; skipping')
            continue

        gt_img = nib.load(lab_path)
        pred_img = nib.load(pred_path)

        voxel_vol_mm3 = float(np.prod(gt_img.header.get_zooms()))

        gt_vol = volume_ml(gt_img.get_fdata(), voxel_vol_mm3)
        pred_vol = volume_ml(pred_img.get_fdata(), voxel_vol_mm3)

        subject_id = lab_name.replace('_lab.nii.gz', '')
        records.append((subject_id, gt_vol, pred_vol))

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    summary_rows = []

    for model_name, results_path in MODELS.items():
        print(f'\n=== {model_name} ===')
        print(f'    Folder: {results_path}')

        records = collect_volumes(results_path)

        if len(records) < 2:
            print(f'  Not enough subjects ({len(records)}) to compute Spearman correlation.')
            continue

        gt_vols = [r[1] for r in records]
        pred_vols = [r[2] for r in records]

        rho, pvalue = spearmanr(gt_vols, pred_vols)

        print(f'  N subjects : {len(records)}')
        print(f'  Spearman rho: {rho:.4f}')
        print(f'  p-value     : {pvalue:.4e}')

        summary_rows.append([model_name, len(records), f'{rho:.4f}', f'{pvalue:.4e}'])

        # ------ Per-subject CSV for this model ------
        safe_name = model_name.lower().replace(' ', '_').replace('-', '_')
        per_subj_csv = os.path.join(OUT_DIR, f'volumes_{safe_name}.csv')
        with open(per_subj_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Subject', 'GT_Volume_ml', 'Pred_Volume_ml'])
            writer.writerows(records)
        print(f'  Per-subject volumes saved to: {per_subj_csv}')

    # ------ Summary CSV across all models ------
    summary_csv = os.path.join(OUT_DIR, 'spearman_summary.csv')
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'N', 'Spearman_rho', 'p_value'])
        writer.writerows(summary_rows)

    print(f'\n=== Summary saved to: {summary_csv} ===')


if __name__ == '__main__':
    main()
