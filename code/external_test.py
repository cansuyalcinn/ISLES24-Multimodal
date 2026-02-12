import os
import argparse
from glob import glob
import h5py
import numpy as np
import torch
import shutil
import SimpleITK as sitk
from tqdm import tqdm

from test_3D_util import test_single_case, calculate_metric_percase
from networks import unet_3D, UNet3D_withClinical, UNet3D_withClinical_DAFT


def load_clinical_map(clinical_file):
    clinical_map = None
    if clinical_file is not None and os.path.exists(clinical_file):
        print(f"Loading clinical data from {clinical_file}")
        try:
            import pandas as pd
            df = pd.read_excel(clinical_file)
            if 'patient_id' in df.columns:
                df = df.set_index('patient_id')
            clinical_map = {str(idx): row.values.astype(np.float32) for idx, row in df.iterrows()}
        except Exception as e:
            print(f"Failed to load clinical file: {e}")
            clinical_map = None
    return clinical_map


def build_net(clinical=False, daft=False, num_classes=2, in_channels=5, clinical_dim=0):
    if clinical:
        if daft:
            net = UNet3D_withClinical_DAFT(n_classes=num_classes, in_channels=in_channels, clinical_in_features=max(1, clinical_dim))
        else:
            net = UNet3D_withClinical(n_classes=num_classes, in_channels=in_channels, clinical_in_features=max(1, clinical_dim))
    else:
        net = unet_3D(n_classes=num_classes, in_channels=in_channels)
    return net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, default="/media/cansu/DiskSpace/Cansu/ISLES24/ISLES24-Multimodal/data/challenge_test_set/test/h5_files_preprocessed"
                        ,help='Directory with test H5 files (preprocessed)')
    parser.add_argument('--model_root', type=str, default="/media/cansu/DiskSpace/Cansu/ISLES24/ISLES24-Multimodal/model/ISLES24-Unet",
                         help='Root directory containing Fold_0..Fold_N model folders')
    parser.add_argument('--out_dir', type=str, default="/media/cansu/DiskSpace/Cansu/ISLES24/ISLES24-Multimodal/data/challenge_test_set/test/results", 
                        help='Output directory to save ensembled predictions')
    
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds to ensemble')
    parser.add_argument('--gpu', type=str, default='0', help='CUDA visible devices')
    parser.add_argument('--clinical', action='store_true', help='Use clinical-enabled models')
    parser.add_argument('--daft', action='store_true', help='Use DAFT clinical models')
    parser.add_argument('--clinical_file', type=str, default=None, help='Path to clinical excel file for mapping')
    parser.add_argument('--seed', type=int, default=1337, help='Seed folder name under each fold, e.g. seed_1337')
    parser.add_argument('--in_channels', type=int, default=5, help='Number of input channels expected by the model')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--patch_size', type=int, nargs=3, default=(96,96,96), help='Patch size for inference')
    parser.add_argument('--stride_xy', type=int, default=64, help='Patch stride xy')
    parser.add_argument('--stride_z', type=int, default=64, help='Patch stride z')

    FLAGS = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    os.makedirs(FLAGS.out_dir, exist_ok=True)

    # find test files
    test_files = sorted(glob(os.path.join(FLAGS.test_dir, '*.h5')))
    if len(test_files) == 0:
        raise FileNotFoundError(f'No .h5 files found in {FLAGS.test_dir}')

    # load clinical map if needed
    clinical_map = None
    clinical_dim = 0
    if FLAGS.clinical:
        clinical_map = load_clinical_map(FLAGS.clinical_file)
        if clinical_map is None:
            print('No clinical map loaded, proceeding without clinical data.')
            clinical_dim = 0
        else:
            # take the first vector length as dimension
            clinical_dim = len(next(iter(clinical_map.values())))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load all fold networks
    nets = []
    for f in range(FLAGS.n_folds):
        fold_dir = os.path.join(FLAGS.model_root, f'Fold_{f}', f'seed_{FLAGS.seed}')
        model_path = os.path.join(fold_dir, 'best_model.pth')
        if not os.path.exists(model_path):
            # try fallback to iter_*.pth last snapshot
            candidates = sorted(glob(os.path.join(fold_dir, '*.pth')))
            if len(candidates) == 0:
                raise FileNotFoundError(f'No model weights found for fold {f} in {fold_dir}')
            model_path = candidates[-1]

        net = build_net(clinical=FLAGS.clinical, daft=FLAGS.daft,
                        num_classes=FLAGS.num_classes, in_channels=FLAGS.in_channels, clinical_dim=clinical_dim)
        net.to(device)
        state = torch.load(model_path, map_location=device)
        # support if saved dict has 'state_dict' or raw state
        if isinstance(state, dict) and 'state_dict' in state and not any(k.startswith('module.') for k in state.keys()):
            # older saver
            state = state['state_dict']
        # handle DataParallel saved keys
        new_state = {}
        if isinstance(state, dict):
            for k, v in state.items():
                new_key = k.replace('module.', '')
                new_state[new_key] = v
            try:
                net.load_state_dict(new_state)
            except Exception as e:
                # last resort: try load whole object
                print(f'Warning loading state_dict for fold {f}: {e}')
                net.load_state_dict(state)
        else:
            # state is not dict: try direct load
            net.load_state_dict(state)

        net.eval()
        nets.append(net)
        print(f'Loaded model for fold {f} from {model_path}')

    total_metric = np.zeros((FLAGS.num_classes-1, 5))

    # iterate cases
    with open(os.path.join(FLAGS.out_dir, 'ensemble_metrics.txt'), 'w') as fout:
        for image_path in tqdm(test_files, desc='Images'):
            ids = os.path.basename(image_path).replace('.h5', '')
            print(f'Processing case {ids}...')

            with h5py.File(image_path, 'r') as h5f:
                image = h5f['data'][:]
                label = h5f['label'][:]

            pid = ids.split('_')[0]
            if FLAGS.clinical:
                cv = clinical_map.get(pid, None) if clinical_map is not None else None
            else:
                cv = None

            # accumulate probability maps
            prob_sum = None
            for net in nets:
                # ensure net on correct device
                net.to(device)
                with torch.no_grad():
                    pred_label, score_map = test_single_case(net, image, FLAGS.stride_xy, 
                                                             FLAGS.stride_z, tuple(FLAGS.patch_size), 
                                                             num_classes=FLAGS.num_classes, clinical_vec=cv)
                if prob_sum is None:
                    prob_sum = score_map.astype(np.float32)
                else:
                    prob_sum += score_map.astype(np.float32)

            # average probabilities
            avg_prob = prob_sum / float(len(nets))
            # predicted label
            pred = np.argmax(avg_prob, axis=0).astype(np.uint8)

            metric = calculate_metric_percase(pred == 1, label == 1)
            total_metric[0, :] += metric

            # change the shape of the pred and label arrays from nibabel (i,j,k) to SimpleITK (k,j,i) for saving
            pred = np.transpose(pred, (2, 1, 0))
            label_save = np.transpose(label.astype(np.uint8), (2, 1, 0))

            fout.writelines(f"{ids},{metric[0]},{metric[1]},{metric[2]},{metric[3]}\n")

            original_ncct_path = f"/media/cansu/DiskSpace/Cansu/ISLES24/ISLES24-Multimodal/data/challenge_test_set/test/raw_data/{pid}/ses-01"
            ncct_img = sitk.ReadImage(os.path.join(original_ncct_path, f'{pid}_ses-01_ncct.nii.gz'))
            # use the original spacing and origin
            spacing = ncct_img.GetSpacing()
            origin = ncct_img.GetOrigin()
            direction = ncct_img.GetDirection()

            # save outputs
            pred_itk = sitk.GetImageFromArray(pred.astype(np.uint8))
            pred_itk.SetSpacing(spacing)
            pred_itk.SetOrigin(origin)
            pred_itk.SetDirection(direction)
            sitk.WriteImage(pred_itk, os.path.join(FLAGS.out_dir, f"{ids}_pred.nii.gz"))

            lab_itk = sitk.GetImageFromArray(label_save)
            lab_itk.SetSpacing(spacing)
            lab_itk.SetOrigin(origin)
            lab_itk.SetDirection(direction)
            sitk.WriteImage(lab_itk, os.path.join(FLAGS.out_dir, f"{ids}_lab.nii.gz"))

            # # img_itk = sitk.GetImageFromArray(image)
            # # img_itk.SetSpacing(spacing)
            # # img_itk.SetOrigin(origin)
            # # img_itk.SetDirection(direction)
            # # sitk.WriteImage(img_itk, os.path.join(FLAGS.out_dir, f"{ids}_img.nii.gz"))

            # save average probability map
            # # np.savez_compressed(os.path.join(FLAGS.out_dir, f"{ids}.npz"), probabilities=avg_prob.astype(np.float32))

        # write mean
        mean_metrics = (total_metric[0, :] / len(test_files))
        fout.writelines(f"Mean metrics,{mean_metrics[0]},{mean_metrics[1]},{mean_metrics[2]},{mean_metrics[3]}\n")

    print('Ensembled testing finished. Results in', FLAGS.out_dir)


if __name__ == '__main__':
    main()