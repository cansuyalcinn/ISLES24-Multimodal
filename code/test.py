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

def Inference(FLAGS):

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    num_classes = 2

    if FLAGS.read_mode == 'test_files.txt':
        mode = 'test'
    elif FLAGS.read_mode == 'val_files.txt':
        mode = 'val'
    elif FLAGS.read_mode == 'train_files.txt':
        mode = 'train'
    else:
        raise ValueError('read_mode should be test_files.txt, val_files.txt or train_files.txt')

    snapshot_path = "../model/{}/".format(FLAGS.exp)
    test_save_path = "../model/{}/Prediction_{}".format(FLAGS.exp, mode)

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
        
    os.makedirs(test_save_path)

    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')

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
            net = UNet3D_withClinical_DAFT(n_classes=num_classes, in_channels=6, clinical_in_features=max(1, clinical_dim)).cuda()
        else:
            net = UNet3D_withClinical(n_classes=num_classes, in_channels=6, clinical_in_features=max(1, clinical_dim)).cuda()

    else:
        net = unet_3D(n_classes=num_classes, in_channels=6).cuda()

    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, base_dir=FLAGS.root_path, test_list=FLAGS.read_mode, num_classes=num_classes,
                               patch_size=(96, 96, 96), stride_xy=64, stride_z=64, test_save_path=test_save_path, model=FLAGS.model,
                               clinical=FLAGS.clinical, clinical_map=clinical_map, clinical_file=FLAGS.clinical_file)
    return avg_metric, mode


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric, mode = Inference(FLAGS)
    test_save_path = "../model/{}".format(FLAGS.exp)
    # save it as a txt file
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    with open(os.path.join(test_save_path, f'metric_{mode}.txt'), 'w') as f:
        f.write('Dice: {}\n'.format(metric))

    print(metric)