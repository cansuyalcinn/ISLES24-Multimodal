import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import random

class ISLES24(Dataset):
    """ ISLES24 Dataset

    Now supports an optional `fold` argument. If `fold` is provided the
    dataset will look for split files named `fold_{fold}_{split}_files.txt`
    inside the `splits/` folder. If the fold-specific file does not exist
    it falls back to the legacy `{split}_files.txt`.
    """

    def __init__(self, base_dir=None, split='train', transform=None, fold: int = None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        # build candidate path for split files
        splits_dir = os.path.join(self._base_dir, 'splits')
        if fold is not None:
            candidate = os.path.join(splits_dir, f'fold_{fold}_{split}_files.txt')
            if os.path.exists(candidate):
                path = candidate
                print(f"Using fold {fold} split file: {os.path.basename(path)}")
            else:
                # fallback to legacy name
                path = os.path.join(splits_dir, f'{split}_files.txt')
                print(f"Fold-specific file not found ({candidate}). Falling back to {os.path.basename(path)}")
        else:
            path = os.path.join(splits_dir, f'{split}_files.txt')

        if not os.path.exists(path):
            raise FileNotFoundError(f"Split file not found: {path}")

        with open(path, 'r') as f:
            self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '').split(',')[0] for item in self.image_list]
        print("Total {} samples in {} set.".format(len(self.image_list), split))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        
        if isinstance(idx, str):
            image_name = idx
        else:
            image_name = self.image_list[idx]

        # open h5 file safely and read datasets
        h5_path = os.path.join(self._base_dir, "h5_files_preprocessed_no_znorm", image_name)
        with h5py.File(h5_path, 'r') as h5f:
            # support files that use either 'data' or 'image' as the image dataset name
            if 'data' in h5f:
                image = h5f['data'][:]
            elif 'image' in h5f:
                image = h5f['image'][:]
            else:
                raise KeyError(f"No 'data' or 'image' dataset found in {h5_path}")

            if 'label' in h5f:
                label = h5f['label'][:]
            elif 'gt' in h5f:
                label = h5f['gt'][:]
            else:
                raise KeyError(f"No 'label' or 'gt' dataset found in {h5_path}")

        # extract patient id from filename (robust to full paths)
        base_name = os.path.basename(image_name)
        patient_id = base_name.split('_')[0] if isinstance(base_name, str) else None

        sample = {'image': image, 'label': label.astype(np.uint8), 'patient_id': patient_id}
        if self.transform:
            sample = self.transform(sample)
        # keep original idx for backward compatibility
        sample["idx"] = idx

        return sample
    

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}



class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # rotate only spatial axes (H, W) for image and (H, W) for label
        k = np.random.randint(0, 4)
        image = np.rot90(image, k, axes=(2, 3))
        label = np.rot90(label, k, axes=(1, 2))

        # flip along one of the spatial axes (depth, height, width)
        flip_axis = np.random.randint(0, 3)
        if flip_axis == 0:
            # flip depth axis
            image = np.flip(image, axis=1).copy()
            label = np.flip(label, axis=0).copy()
        elif flip_axis == 1:
            # flip height axis
            image = np.flip(image, axis=2).copy()
            label = np.flip(label, axis=1).copy()
        else:
            # flip width axis
            image = np.flip(image, axis=3).copy()
            label = np.flip(label, axis=2).copy()

        return {'image': image, 'label': label}

class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(
            image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}

class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros(
            (self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}
    
import numpy as np

class RandomCrop(object):
    """
    Randomly crop a 3D subvolume from (C, D, H, W) image and (D, H, W) label.
    If p_foreground > 0, with that probability the crop will be sampled so that
    it contains at least one foreground voxel (label > 0). Otherwise the crop is
    chosen uniformly at random. If no foreground exists in the label, falls
    back to a random crop.
    """
    def __init__(self, output_size, with_sdf=False, p_foreground=0.5, foreground_key='label'):
        assert len(output_size) == 3, "output_size should be a 3D tuple (D, H, W)"
        self.output_size = tuple(output_size)
        self.with_sdf = with_sdf
        self.p_foreground = float(p_foreground)
        self.foreground_key = foreground_key

    def __call__(self, sample):
        image, label = sample['image'], sample[self.foreground_key]
        # image: (C,D,H,W), label: (D,H,W)

        assert image.ndim == 4, f"Expected image shape (C,D,H,W), got {image.shape}"
        assert label.ndim == 3, f"Expected label shape (D,H,W), got {label.shape}"

        C, D, H, W = image.shape
        out_D, out_H, out_W = self.output_size

        # pad if smaller than desired
        pad_D = max(out_D - D, 0)
        pad_H = max(out_H - H, 0)
        pad_W = max(out_W - W, 0)

        if pad_D > 0 or pad_H > 0 or pad_W > 0:
            pad_before = (pad_D // 2, pad_H // 2, pad_W // 2)
            pad_after = (pad_D - pad_before[0], pad_H - pad_before[1], pad_W - pad_before[2])

            image = np.pad(image,
                           ((0, 0), (pad_before[0], pad_after[0]),
                            (pad_before[1], pad_after[1]),
                            (pad_before[2], pad_after[2])),
                           mode='constant')
            label = np.pad(label,
                           ((pad_before[0], pad_after[0]),
                            (pad_before[1], pad_after[1]),
                            (pad_before[2], pad_after[2])),
                           mode='constant')

            if self.with_sdf and 'sdf' in sample:
                sdf = np.pad(sample['sdf'],
                             ((pad_before[0], pad_after[0]),
                              (pad_before[1], pad_after[1]),
                              (pad_before[2], pad_after[2])),
                             mode='constant')

        # update shapes after padding
        _, D, H, W = image.shape

        # Decide whether to force a foreground-containing crop
        do_foreground = (np.random.rand() < self.p_foreground)

        if do_foreground:
            # find foreground voxel indices
            fg_coords = np.where(label > 0)
            if len(fg_coords[0]) == 0:
                # no foreground present, fallback to random crop
                do_foreground = False

        if do_foreground:
            # pick a random foreground voxel and place crop so it is inside the crop
            idx = np.random.randint(0, len(fg_coords[0]))
            zf, yf, xf = int(fg_coords[0][idx]), int(fg_coords[1][idx]), int(fg_coords[2][idx])

            # compute valid start ranges so the chosen voxel is inside [d1, d1+out_D-1]
            d1_min = max(0, zf - out_D + 1)
            d1_max = min(zf, D - out_D)
            h1_min = max(0, yf - out_H + 1)
            h1_max = min(yf, H - out_H)
            w1_min = max(0, xf - out_W + 1)
            w1_max = min(xf, W - out_W)

            # handle edge cases where min > max by clipping
            if d1_min > d1_max:
                d1_min = max(0, D - out_D)
                d1_max = d1_min
            if h1_min > h1_max:
                h1_min = max(0, H - out_H)
                h1_max = h1_min
            if w1_min > w1_max:
                w1_min = max(0, W - out_W)
                w1_max = w1_min

            d1 = np.random.randint(d1_min, d1_max + 1)
            h1 = np.random.randint(h1_min, h1_max + 1)
            w1 = np.random.randint(w1_min, w1_max + 1)
        else:
            # uniform random crop
            d1 = np.random.randint(0, D - out_D + 1)
            h1 = np.random.randint(0, H - out_H + 1)
            w1 = np.random.randint(0, W - out_W + 1)

        # crop
        image = image[:, d1:d1+out_D, h1:h1+out_H, w1:w1+out_W]
        label = label[d1:d1+out_D, h1:h1+out_H, w1:w1+out_W]

        result = {'image': image, 'label': label}
        if self.with_sdf and 'sdf' in sample:
            sdf = sdf[d1:d1+out_D, h1:h1+out_H, w1:w1+out_W]
            result['sdf'] = sdf

        return result

import torch

class ToTensor(object):
    """Convert ndarrays in sample to PyTorch Tensors."""

    def __call__(self, sample):
        # ensure correct dtypes and produce contiguous, cloned tensors so
        # PyTorch's collate can safely resize/stack them across workers
        image = sample['image'].astype(np.float32)
        label = sample['label'].astype(np.int64)

        img_t = torch.from_numpy(image).contiguous().float().clone()
        lbl_t = torch.from_numpy(label).long().contiguous().clone()

        result = {'image': img_t, 'label': lbl_t}
        if 'sdf' in sample:
            result['sdf'] = torch.from_numpy(sample['sdf'].astype(np.float32)).contiguous().float().clone()
        if 'onehot_label' in sample:
            result['onehot_label'] = torch.from_numpy(sample['onehot_label'].astype(np.int64)).long().contiguous().clone()

        return result

# A new implementation of random rotation and flip from https://github.com/youngyzzZ/SSL-w2sPC/blob/main/src/dataloaders/dataset.py

def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample