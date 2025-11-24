import argparse
import os
import shutil
from glob import glob
import torch
from test_3D_util import test_all_case
from networks import unet_3D, UNet3D_withClinical, UNet3D_withClinical_DAFT
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/media/cansu/DiskSpace/Cansu/ISLES24/ISLES24-Multimodal/data', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='ISLES24-Unet', help='experiment_name')
parser.add_argument('--gpu', type=str, default='0', help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--read_mode', type=str, default='test_files.txt', help='what to do test on')
parser.add_argument('--model', type=str, default='unet_3D', help='model_name')
parser.add_argument('--clinical', action='store_true', help='Enable clinical model evaluation (UNet3D_withClinical)')
parser.add_argument('--daft', action='store_true', help='Enable DAFT model evaluation (UNet3D_withClinical_DAFT)')
parser.add_argument('--clinical_file', type=str, default=None, help='Path to clinical_tabular_processed.xlsx -- /media/cansu/DiskSpace/Cansu/ISLES24/ISLES24-Multimodal/data/clinical_tabular_processed.xlsx')
parser.add_argument('--fold', type=str, default=None, help='Cross-validation fold index (0..4). If provided uses fold-specific split files.')
parser.add_argument('--seed', type=int,  default=1337, help='random seed for the model setting but for the data we use a different seed.')

def Inference(FLAGS):

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    num_classes = 2
    # ensure clinical_map is always defined so it's safe to pass to test_all_case
    clinical_map = None

    mode = 'test'
    # Determine which split list to use. If a fold is provided, look for fold_{fold}_{read_mode}
    if FLAGS.fold is not None:
        test_files = f'fold_{FLAGS.fold}_{FLAGS.read_mode}'
        snapshot_path = "../model/{}/Fold_{}/seed_{}".format(FLAGS.exp, FLAGS.fold, FLAGS.seed)
        test_save_path = "../model/{}/Fold_{}/seed_{}/Prediction_{}".format(FLAGS.exp, FLAGS.fold, FLAGS.seed, mode)
    else:
        test_files = FLAGS.read_mode
        # when fold not provided, fall back to generic model folder
        snapshot_path = "../model/{}/".format(FLAGS.exp)
        test_save_path = "../model/{}/Prediction_{}".format(FLAGS.exp, mode)

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
        
    os.makedirs(test_save_path)

    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')

    # Verify split file exists under the provided root path
    split_path = os.path.join(FLAGS.root_path, 'splits', test_files)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}. Please create splits or pass --read_mode accordingly.")

    if FLAGS.clinical:
        # load clinical mapping to determine input dim
        clinical_dim = 0
        clinical_map = None
        if FLAGS.clinical_file is not None and os.path.exists(FLAGS.clinical_file):
            try:
                df = pd.read_excel(FLAGS.clinical_file)
                if 'patient_id' in df.columns:
                    df = df.set_index('patient_id')
                clinical_dim = df.shape[1]
                clinical_map = {str(idx): row.values.astype(np.float32) for idx, row in df.iterrows()}
            except Exception as e:
                print(f"Failed to load clinical file: {e}")
                clinical_map = None

        if FLAGS.daft:
            net = UNet3D_withClinical_DAFT(n_classes=num_classes, in_channels=5, clinical_in_features=max(1, clinical_dim)).cuda()
        else:
            net = UNet3D_withClinical(n_classes=num_classes, in_channels=5, clinical_in_features=max(1, clinical_dim)).cuda()

    else:
        net = unet_3D(n_classes=num_classes, in_channels=5).cuda()

    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, base_dir=FLAGS.root_path, test_list=test_files, num_classes=num_classes,
                               patch_size=(96, 96, 96), stride_xy=64, stride_z=64, test_save_path=test_save_path, model=FLAGS.model,
                               clinical=FLAGS.clinical, clinical_map=clinical_map, clinical_file=FLAGS.clinical_file)
    return avg_metric, mode


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric, mode = Inference(FLAGS)
    test_save_path = "../model/{}/Fold_{}/seed_{}/".format(FLAGS.exp, FLAGS.fold, FLAGS.seed)
    # save it as a txt file
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    with open(os.path.join(test_save_path, f'metric_{mode}.txt'), 'w') as f:
        f.write('Dice: {}\n'.format(metric))

    print(metric)