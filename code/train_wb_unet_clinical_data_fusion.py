import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# removed mixed-precision (autocast/GradScaler) for simpler deterministic training
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from dataset import (ISLES24, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor)
from networks import UNet3D_withClinical
import pandas as pd
# from val_3D import test_all_case
# not using AMP autocast -> remove GradScaler/amp usage
from utils import DiceLoss
from val_3D import test_all_case

# Optional Weights & Biases (wandb) integration
try:
    import wandb
    _WANDB_AVAILABLE = True
    print("Weights & Biases (wandb) is available.")
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False
    print("Weights & Biases (wandb) is not available.")

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/media/cansu/DiskSpace/Cansu/ISLES24/ISLES24-Multimodal/data', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='ISLES24-Unet_DF', help='experiment_name')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[96, 96, 96], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed for the model setting but for the data we use a different seed.')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--fold', type=int, default=None, help='Cross-validation fold index (0..4). If provided uses fold-specific split files.')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def _tensor_to_image(grid_tensor):
    # grid_tensor: CPU tensor [C,H,W], values in [0,1] or arbitrary; convert to HWC uint8
    grid_np = grid_tensor.detach().cpu().numpy()
    if grid_np.ndim == 3:
        grid_np = np.transpose(grid_np, (1, 2, 0))  # HWC
    else:
        grid_np = grid_np
    # normalize to 0-255
    grid_min = grid_np.min()
    grid_max = grid_np.max()
    if grid_max > grid_min:
        grid_np = (grid_np - grid_min) / (grid_max - grid_min)
    grid_np = (grid_np * 255.0).astype(np.uint8)
    return grid_np

def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 2
    torch.manual_seed(args.seed)

    # --- load preprocessed clinical tabular data ---
    clinical_file = os.path.join(train_data_path, 'clinical_tabular_processed.xlsx')
    if os.path.exists(clinical_file):
        print(f"Loading clinical data from {clinical_file}...")
        try:
            clin_df = pd.read_excel(clinical_file)
            if 'patient_id' in clin_df.columns:
                clin_df = clin_df.set_index('patient_id')

            # detect mask columns (those ending with '_mask') and value columns
            all_cols = list(clin_df.columns)
            mask_cols = [c for c in all_cols if str(c).endswith('_mask')]
            value_cols = [c for c in all_cols if c not in mask_cols]
            n_value_feats = len(value_cols)
            clinical_dim = len(all_cols)

            # build a mapping patient_id -> numpy array (values followed by masks)
            clinical_map = {str(idx): row.values.astype(np.float32) for idx, row in clin_df.iterrows()}
        except Exception as e:
            print(f"Failed to load clinical file '{clinical_file}': {e}")
            clinical_map = {}
            clinical_dim = 0
    else:
        print(f"Clinical file not found at {clinical_file}. Proceeding without clinical features (zeros).")
        clinical_map = {}
        clinical_dim = 0

    model = UNet3D_withClinical(in_channels=5, n_classes=num_classes, clinical_in_features=max(1, clinical_dim))

    db_train = ISLES24(base_dir=train_data_path,
                         split='train',
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ]),
                         fold=args.fold)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(2)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    model.cuda()
    # mixed precision removed: use standard FP32 training

    # Optional: watch model gradients and parameters with wandb
    if _WANDB_AVAILABLE:
        try:
            wandb.watch(model, log='all', log_freq=100)
        except Exception:
            pass

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            # --- prepare clinical vector batch ---
            # sampled_batch['idx'] may be a list/tensor of indices or filenames
            idx_field = sampled_batch.get('idx', None)
            batch_filenames = []
            if idx_field is None:
                # fallback: try to recover from trainloader image_list using i_batch
                batch_filenames = [db_train.image_list[i_batch]]
            else:
                # normalize to python list
                if isinstance(idx_field, torch.Tensor):
                    idx_list = idx_field.cpu().tolist()
                else:
                    idx_list = idx_field
                # if idx_list is a scalar make it a list
                if not isinstance(idx_list, (list, tuple)):
                    idx_list = [idx_list]
                for it in idx_list:
                    if isinstance(it, int):
                        fn = db_train.image_list[it]
                    else:
                        fn = str(it)
                    # ensure we only have the filename (not full path)
                    # some entries might be 'sub-stroke0001_ses-01_all_modalities.h5'
                    batch_filenames.append(os.path.basename(fn))

            # derive patient ids from filenames (assumes 'sub-strokeXXXX_...' pattern)
            clinical_batch_list = []
            for fn in batch_filenames:
                pid = fn.split('_')[0] if fn is not None else None

                if pid is not None and pid in clinical_map and clinical_dim > 0: # found clinical data
                    arr = clinical_map[pid]  # numpy array length clinical_dim: [values..., masks...]
                    vals = torch.from_numpy(arr[:n_value_feats].copy()).float()
                    masks = torch.from_numpy(arr[n_value_feats:].copy()).float() if (clinical_dim - n_value_feats) > 0 else torch.ones_like(vals)

                    # only observed features may be knocked out
                    observed_idx = torch.where(masks == 1.0)[0]
                    num_obs = len(observed_idx)

                    if num_obs > 0:
                        # sample per-patient dropout ratio
                        p = torch.rand(1).item()  # random number in [0,1]
                        
                        # compute number of features to drop
                        num_drop = max(1, int(p * num_obs))  # drop at least 1
                        
                        # randomly pick exactly num_drop features among observed ones
                        # torch.randperm(n) generates a random permutation of the integers from 0 to n-1, with no repeats. shuffle the indices
                        perm = torch.randperm(num_obs)

                        drop_idx = observed_idx[perm[:num_drop]]

                        # apply knockout
                        vals[drop_idx] = -10.0
                        masks[drop_idx] = 0.0

                    combined = torch.cat([vals, masks], dim=0)
                    clinical_batch_list.append(combined)
                else:
                    # fallback zero vector if missing or clinical not available
                    clinical_batch_list.append(torch.zeros(max(1, clinical_dim), dtype=torch.float32))

            clinical_batch = torch.stack(clinical_batch_list, dim=0).cuda()

            optimizer.zero_grad()

            outputs = model(volume_batch, clinical_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            # loss_ce = ce_loss(outputs, label_batch)
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = loss_dice

            # backprop and optimizer step (FP32)
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            # log to wandb if available
            if _WANDB_AVAILABLE:
                try:
                    wandb.log({
                        'lr': lr_,
                        'train/loss_dice': loss_dice.item()
                    }, step=iter_num)
                except Exception:
                    pass

            logging.info(
                'iteration %d : loss_dice: %f' %
                (iter_num, loss_dice.item()))
            writer.add_scalar('loss/loss_dice', loss_dice, iter_num)

            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_pred = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_pred, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_label = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_label, iter_num)

                if _WANDB_AVAILABLE:
                    try:
                        wandb.log({
                            'train/Image': wandb.Image(_tensor_to_image(grid_image)),
                            'train/Predicted_label': wandb.Image(_tensor_to_image(grid_pred)),
                            'train/Groundtruth_label': wandb.Image(_tensor_to_image(grid_label))
                        }, step=iter_num)
                    except Exception:
                        pass

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                # prepare clinical map for the validator (convert tensors to numpy)
                try:
                    clinical_map_np = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else np.array(v)) for k, v in clinical_map.items()}
                except Exception:
                    clinical_map_np = clinical_map if clinical_map is not None else None
                    
                test_list = f'fold_{args.fold}_val_files.txt' if args.fold is not None else 'val_files.txt'
                avg_metric = test_all_case(model, args.root_path, test_list=test_list, num_classes=2, 
                                           patch_size=args.patch_size, stride_xy=64, stride_z=64, clinical=True, clinical_map=clinical_map_np)
                
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                    if _WANDB_AVAILABLE:
                        try:
                            wandb.save(save_mode_path)
                            wandb.save(save_best)
                        except Exception:
                            pass

                writer.add_scalar('info/val_dice_score',avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_hd95', avg_metric[0, 1], iter_num)
                logging.info('iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))

                if _WANDB_AVAILABLE:
                    try:
                        wandb.log({
                            'val/dice_score': float(avg_metric[0, 0].mean()),
                            'val/hd95': float(avg_metric[0, 1].mean())
                        }, step=iter_num)
                    except Exception:
                        pass

                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                if _WANDB_AVAILABLE:
                    try:
                        wandb.save(save_mode_path)
                    except Exception:
                        pass

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # include fold and seed in snapshot path so logs/models are separated per run
    snapshot_path = "../model/{}/Fold_{}/seed_{}".format(args.exp, args.fold, args.seed)
    snapshot_path = os.path.join(snapshot_path)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # Initialize Weights & Biases (optional)
    if _WANDB_AVAILABLE:
        try:
            wandb.init(project=args.exp, name=f"{args.exp}_fold{args.fold}_seed{args.seed}", config=vars(args), reinit=True)
        except Exception:
            pass

    train(args, snapshot_path)

    if _WANDB_AVAILABLE:
        try:
            wandb.finish()
        except Exception:
            pass