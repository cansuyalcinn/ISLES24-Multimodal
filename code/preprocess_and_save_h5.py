#!/usr/bin/env python3
"""Preprocess ISLES24 raw/derivative images, save per-modality NIfTIs and combined HDF5 files.

This script implements the histogram equalization with custom intensity ranges
for the perfusion/CTA modalities and saves:

- per-modality NIfTI files under data/preprocessed_data/<modality>/
- combined HDF5 files under data/h5_files_preprocessed/ with datasets 'data' and 'label'

Usage example:
    python code/preprocess_and_save_h5.py --paths_raw /media/cansu/DiskSpace/Cansu/ISLES24/train/raw_data \
        --paths_derivatives /media/cansu/DiskSpace/Cansu/ISLES24/train/derivatives \
        --out_dir /media/cansu/DiskSpace/Cansu/ISLES24/ISLES24-Multimodal/data
"""
import os
import sys
import argparse
import glob
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple

import nibabel as nib
import numpy as np
from skimage import exposure
import h5py


def apply_histogram_equalization_custom_range(input_file: str, output_file: str, min_intensity: float, max_intensity: float) -> None:
    img = nib.load(input_file)
    data = img.get_fdata()

    # Clip to provided range
    data_clipped = np.clip(data, min_intensity, max_intensity)
    # keep background zeros
    data_clipped[data == 0] = 0

    # normalize to [0,1]
    denom = (max_intensity - min_intensity)
    if denom == 0:
        denom = 1.0
    mask_pos = data_clipped > 0.0001
    data_clipped[mask_pos] = data_clipped[mask_pos] - min_intensity
    data_normalized = np.zeros_like(data_clipped, dtype=np.float32)
    data_normalized[mask_pos] = data_clipped[mask_pos] / float(denom)

    # apply histogram equalization only on positive region
    with np.errstate(all='ignore'):
        equalized = exposure.equalize_hist(data_normalized, mask=mask_pos)

    # set non-positive back to zero
    equalized[~mask_pos] = 0.0

    # save as new nifti using original affine/header
    out_img = nib.Nifti1Image(equalized.astype(np.float32), img.affine, img.header)
    nib.save(out_img, output_file)


def process_patient(patient_name: str, paths_raw: str, paths_deriv: str, out_dir: str, intensity_ranges: Dict[str, Tuple[float, float]]) -> Tuple[str, bool, str]:
    """Process one patient: preprocess modalities and save combined HDF5.

    Returns (patient_name, success, message)
    """
    try:
        # build expected file paths
        # modalities we will include in the HDF5 (order matters)
        modalities = [
            ('cbf', os.path.join(paths_deriv, patient_name, 'ses-01', 'perfusion-maps', f'{patient_name}_ses-01_space-ncct_cbf.nii.gz')),
            ('cbv', os.path.join(paths_deriv, patient_name, 'ses-01', 'perfusion-maps', f'{patient_name}_ses-01_space-ncct_cbv.nii.gz')),
            ('mtt', os.path.join(paths_deriv, patient_name, 'ses-01', 'perfusion-maps', f'{patient_name}_ses-01_space-ncct_mtt.nii.gz')),
            ('tmax', os.path.join(paths_deriv, patient_name, 'ses-01', 'perfusion-maps', f'{patient_name}_ses-01_space-ncct_tmax.nii.gz')),
            ('cta', os.path.join(paths_deriv, patient_name, 'ses-01', f'{patient_name}_ses-01_space-ncct_cta.nii.gz')),
        ]
        
        if args.test_set:
            gt_path = os.path.join(paths_deriv, patient_name, 'ses-02', f'{patient_name}_ses-02_ncct-masked_lesion-msk.nii.gz')
        else:
            gt_path = os.path.join(paths_deriv, patient_name, 'ses-02', f'{patient_name}_ses-02_space-ncct_lesion-msk.nii.gz')

        preproc_root = os.path.join(out_dir, 'preprocessed_data')
        h5_out_root = os.path.join(out_dir, 'h5_files_preprocessed')
        os.makedirs(preproc_root, exist_ok=True)
        os.makedirs(h5_out_root, exist_ok=True)

        # ensure modality subdirs exist
        for mod, _ in modalities:
            os.makedirs(os.path.join(preproc_root, mod), exist_ok=True)

        # process each modality
        modality_arrays = []
        ref_shape = None
        for idx, (mod, path) in enumerate(modalities):
            if not os.path.exists(path):
                return (patient_name, False, f'Missing modality file: {path}')

            out_nii = os.path.join(preproc_root, mod, f'{patient_name}_{mod}.nii.gz')

            # apply histogram equalization with custom ranges if provided
            rng = None
            # mapping by modality name to intensity_ranges keys used in notebook: 0000..0004
            key_map = {'cbf': '0000', 'cbv': '0001', 'mtt': '0002', 'tmax': '0003', 'cta': '0004'}
            if key_map.get(mod) in intensity_ranges:
                rng = intensity_ranges[key_map[mod]]

            if rng is not None:
                # skip processing if already exists
                if not os.path.exists(out_nii):
                    apply_histogram_equalization_custom_range(path, out_nii, rng[0], rng[1])
            else:
                # copy raw file if no range specified
                if not os.path.exists(out_nii):
                    shutil.copy(path, out_nii)

            # load preprocessed nii to array
            nii = nib.load(out_nii)
            arr = nii.get_fdata().astype(np.float32)
            # ensure shape is (D,H,W) or (X,Y,Z) consistent; we expect 3D volumes
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = np.squeeze(arr, axis=0)

            if ref_shape is None:
                ref_shape = arr.shape
            else:
                if arr.shape != ref_shape:
                    return (patient_name, False, f'Shape mismatch for {mod}: {arr.shape} vs {ref_shape}')

            modality_arrays.append(arr)

        # load label
        if not os.path.exists(gt_path):
            return (patient_name, False, f'Missing GT file: {gt_path}')
        gt_nii = nib.load(gt_path)
        gt_arr = gt_nii.get_fdata().astype(np.uint8)

        # ensure label shape matches
        if gt_arr.shape != ref_shape:
            return (patient_name, False, f'Label shape {gt_arr.shape} does not match image shape {ref_shape}')

        # stack modalities into array with channel-first ordering (C, D, H, W)
        data_stack = np.stack(modality_arrays, axis=0).astype(np.float32)

        # write HDF5
        h5_path = os.path.join(h5_out_root, f'{patient_name}_ses-01_all_modalities.h5')
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset('data', data=data_stack, compression='gzip')
            hf.create_dataset('label', data=gt_arr.astype(np.uint8), compression='gzip')

        return (patient_name, True, f'Wrote {h5_path}')

    except Exception as e:
        return (patient_name, False, str(e))


def collect_patient_list(paths_raw: str) -> list:
    # patient list inferred from the raw directory. Accept either patient
    # directories (preferred) or raw files named like <patient>.nii.gz.
    if not os.path.exists(paths_raw):
        return []

    entries = os.listdir(paths_raw)
    patients = []
    for e in entries:
        full = os.path.join(paths_raw, e)
        if os.path.isdir(full):
            patients.append(e)
        elif os.path.isfile(full):
            name = e
            for ext in ('.nii.gz', '.nii', '.tar.gz', '.tgz', '.zip', '.gz'):
                if name.endswith(ext):
                    name = name[: -len(ext)]
            patients.append(name)

    # deduplicate and prefer 'sub-' prefixed ids when available
    patients = sorted(list(set(patients)))
    prefixed = [p for p in patients if p.startswith('sub-')]
    if len(prefixed) > 0:
        return prefixed
    return patients


def main(paths_raw: str, paths_deriv: str, out_dir: str, workers: int = None):
    # intensity ranges mapping used in notebook
    intensity_ranges = {
        '0000': (0, 35),   # CBF
        '0001': (0, 10),   # CBV
        '0002': (0, 20),   # MTT
        '0003': (0, 7),    # TMAX
        '0004': (0, 90),   # CTA
    }

    patients = collect_patient_list(paths_raw)
    if len(patients) == 0:
        print('No patient files found in raw path:', paths_raw)
        sys.exit(1)

    max_workers = workers or min(8, (os.cpu_count() or 1))
    print(f'Processing {len(patients)} patients with {max_workers} workers...')

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_patient, p, paths_raw, paths_deriv, out_dir, intensity_ranges): p for p in patients}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = (p, False, f'Exception: {e}')
            results.append(res)
            name, ok, msg = res
            print(f'[{"OK" if ok else "ERR"}] {name}: {msg}')

    # summary
    n_ok = sum(1 for _, ok, _ in results if ok)
    n_err = len(results) - n_ok
    print(f'Done. Success: {n_ok}, Errors: {n_err}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths_raw', type=str, required=False, default='/media/cansu/DiskSpace/Cansu/ISLES24/train/raw_data')
    parser.add_argument('--paths_derivatives', type=str, required=False, default='/media/cansu/DiskSpace/Cansu/ISLES24/train/derivatives')
    parser.add_argument('--out_dir', type=str, required=False, default='/media/cansu/DiskSpace/Cansu/ISLES24/ISLES24-Multimodal/data')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--test_set', type=bool, default=False, help='test set preprocessing without gt file')
    args = parser.parse_args()
    main(args.paths_raw, args.paths_derivatives, args.out_dir, workers=args.workers)