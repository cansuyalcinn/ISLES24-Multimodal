"""Create K-fold cross-validation split files for the ISLES24 dataset.

This script collects all h5 filenames under `data/h5_files/` (or from
existing split lists if you prefer) and builds K folds. For each fold i
it writes three files into `data/splits/`:

- fold_{i}_train_files.txt
- fold_{i}_val_files.txt
- fold_{i}_test_files.txt

The convention used here is: test = fold_i, val = fold_{(i+1)%K}, train = all
remaining folds.
"""
import argparse
import os
import random
from glob import glob

def main(data_dir: str, k: int = 5, seed: int = 1337):
    # prefer preprocessed h5 files (created by preprocess_and_save_h5.py)
    h5_dir = os.path.join(data_dir, 'h5_files_preprocessed')
    splits_dir = os.path.join(data_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    # collect h5 filenames (just the basename)
    pattern = os.path.join(h5_dir, '*.h5')
    files = [os.path.basename(p) for p in glob(pattern)]
    if len(files) == 0:
        # fallback: if h5_files empty, try collecting from existing split files
        for name in ('train_files.txt', 'val_files.txt', 'test_files.txt'):
            p = os.path.join(splits_dir, name)
            if os.path.exists(p):
                with open(p, 'r') as f:
                    for line in f:
                        fname = line.strip().split(',')[0]
                        if fname not in files:
                            files.append(fname)

    files = sorted(files)
    random.seed(seed)
    random.shuffle(files)

    n = len(files)
    if n == 0:
        raise RuntimeError(f"No files found in {h5_dir} nor in existing split files under {splits_dir}")

    # partition into k folds as evenly as possible
    folds = [[] for _ in range(k)]
    for i, f in enumerate(files):
        folds[i % k].append(f)

    # write splits for each fold
    for i in range(k):
        test_fold = i
        val_fold = (i + 1) % k
        train_fnames = []
        for j in range(k):
            if j not in (test_fold, val_fold):
                train_fnames.extend(folds[j])

        test_fnames = folds[test_fold]
        val_fnames = folds[val_fold]

        train_path = os.path.join(splits_dir, f'fold_{i}_train_files.txt')
        val_path = os.path.join(splits_dir, f'fold_{i}_val_files.txt')
        test_path = os.path.join(splits_dir, f'fold_{i}_test_files.txt')

        with open(train_path, 'w') as f:
            for it in train_fnames:
                f.write(f"{it}\n")
        with open(val_path, 'w') as f:
            for it in val_fnames:
                f.write(f"{it}\n")
        with open(test_path, 'w') as f:
            for it in test_fnames:
                f.write(f"{it}\n")

        print(f"Wrote fold {i}: train={len(train_fnames)}, val={len(val_fnames)}, test={len(test_fnames)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create K-fold cross-validation splits for ISLES24')
    parser.add_argument('--data_dir', type=str, default=os.path.join('..', 'data'), help='Path to dataset root (contains h5_files/ and splits/)')
    parser.add_argument('--k', type=int, default=5, help='Number of folds')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed for shuffling')
    args = parser.parse_args()
    main(args.data_dir, k=args.k, seed=args.seed)
