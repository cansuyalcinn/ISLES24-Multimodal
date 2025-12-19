#!/usr/bin/env python3
"""
Script to evaluate all test files in a results folder and save metrics to CSV.
Columns (order): Dice, AVD (ml), F1 (%), ALCD

Usage:
    python code/evaluate_all.py /path/to/results --out /path/to/output.csv

Assumes ground truth files end with `_lab.nii.gz` and predictions with `_pred.nii.gz`.
"""

import os
import argparse
import csv

import nibabel as nib
import numpy as np

import eval_utils
from tqdm import tqdm


def evaluate_folder(results_path, out_csv):
    files = [f for f in os.listdir(results_path) if f.endswith('_lab.nii.gz')]
    files.sort()

    rows = []

    for f in tqdm(files):
        gt_path = os.path.join(results_path, f)
        pred_path = gt_path.replace('_lab.nii.gz', '_pred.nii.gz')

        if not os.path.exists(pred_path):
            print(f'Warning: prediction not found for {f}, expected {os.path.basename(pred_path)}; skipping')
            continue

        # Load volumes
        gt_img = nib.load(gt_path)
        pred_img = nib.load(pred_path)
        gt = gt_img.get_fdata()
        pred = pred_img.get_fdata()

        # Compute metrics
        f1, alcd, dice = eval_utils.compute_dice_f1_instance_difference(gt, pred)
        # compute_dice_f1_instance_difference returns (f1, instance_count_difference, dice)

        f1_percent = float(f1) * 100.0

        voxel_volume = np.prod(gt_img.header.get_zooms()) / 1000.0
        avd = float(eval_utils.compute_absolute_volume_difference(gt, pred, voxel_volume))

        rows.append([
            f.replace('_lab.nii.gz', ''),  # ID
            float(dice),                    # Dice
            avd,                            # AVD in ml
            f1_percent,                     # F1 in percent
            int(alcd),                      # ALCD
        ])

    # Write CSV
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'Dice', 'AVD_ml', 'F1_percent', 'ALCD'])
        writer.writerows(rows)

    print(f'Wrote {len(rows)} rows to: {out_csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate test set and save metrics to CSV')
    parser.add_argument('--results_path', help='Folder containing *_lab.nii.gz and corresponding *_pred.nii.gz')
    parser.add_argument('--out', default='evaluation_results.csv', help='Output CSV path')
    args = parser.parse_args()

    evaluate_folder(args.results_path, args.out)
