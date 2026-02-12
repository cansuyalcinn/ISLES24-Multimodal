import math

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from skimage.measure import label
from tqdm import tqdm
import os


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1, clinical_vec=None):
    """
    Supports either:
      - image with shape (w, h, d)  [single-channel, legacy]
      - image with shape (C, w, h, d)  [multimodal channels first]
    """
    # detect multimodal channel-first input
    if image.ndim == 4:
        has_channel = True
        channels, w, h, d = image.shape
    elif image.ndim == 3:
        has_channel = False
        channels = 1
        w, h, d = image.shape
    else:
        raise ValueError(f"Unsupported image ndim: {image.ndim}")

    # if the size of image is less than patch_size, then padding it (only spatial dims)
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2

    if add_pad:
        if has_channel:
            image = np.pad(image, [(0, 0), (wl_pad, wr_pad), (hl_pad, hr_pad),
                                   (dl_pad, dr_pad)], mode='constant', constant_values=0)
        else:
            image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                                   (dl_pad, dr_pad)], mode='constant', constant_values=0)

    # spatial sizes after padding
    if has_channel:
        ww, hh, dd = image.shape[1], image.shape[2], image.shape[3]
    else:
        ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((num_classes, ) + (ww, hh, dd)).astype(np.float32)
    cnt = np.zeros((ww, hh, dd)).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                if has_channel:
                    test_patch = image[:, xs:xs+patch_size[0],
                                          ys:ys+patch_size[1],
                                          zs:zs+patch_size[2]]
                    # shape -> (C, px, py, pz), add batch dim -> (1, C, px, py, pz)
                    test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                else:
                    test_patch = image[xs:xs+patch_size[0],
                                      ys:ys+patch_size[1],
                                      zs:zs+patch_size[2]]
                    # legacy: add channel and batch dims -> (1,1,px,py,pz)
                    test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)

                test_patch = torch.from_numpy(test_patch).cuda()

                # prepare clinical tensor if provided
                if clinical_vec is not None:
                    if isinstance(clinical_vec, np.ndarray):
                        c = torch.from_numpy(clinical_vec.astype(np.float32)).unsqueeze(0).cuda()
                    elif isinstance(clinical_vec, torch.Tensor):
                        c = clinical_vec.unsqueeze(0).cuda() if clinical_vec.dim() == 1 else clinical_vec.cuda()
                    else:
                        c = torch.from_numpy(np.array(clinical_vec, dtype=np.float32)).unsqueeze(0).cuda()
                else:
                    c = None

                with torch.no_grad():
                    if c is None:
                        y1 = net(test_patch)
                    else:
                        y1 = net(test_patch, c)
                    y = torch.softmax(y1, dim=1)

                y = y.cpu().data.numpy()
                # y shape expected: (1, num_classes, px, py, pz)
                y = y[0, :, :, :, :]

                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1

    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        # remove spatial padding; keep channel dim out of score_map
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]

    return label_map, score_map

def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return np.array([dice, hd95])
    else:
        return np.zeros(2)


def test_all_case(net, base_dir, test_list="test_list.txt", model = "unet_3D",
                  num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24, 
                  test_save_path=None, model_name=None, clinical=False, clinical_map=None, clinical_file=None,
                  original_img_dir=None):
    
    with open(base_dir + '/splits' + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()

    image_list = [base_dir + "/h5_files_preprocessed/{}".format( item.replace('\n', '').split(",")[0]) for item in image_list]
    
    total_metric = np.zeros((num_classes-1, 5))

    # prepare clinical map if requested
    if clinical:
        if clinical_map is None and clinical_file is not None:
            try:
                import pandas as pd
                df = pd.read_excel(clinical_file)
                if 'patient_id' in df.columns:
                    df = df.set_index('patient_id')
                clinical_map = {str(idx): row.values.astype(np.float32) for idx, row in df.iterrows()}
            except Exception as e:
                print(f"Failed to load clinical file for testing: {e}")
                clinical_map = {}
        elif clinical_map is None:
            clinical_map = {}

    print("Testing begin")

    with open(test_save_path + f"/{model}.txt", "w") as f:
        for image_path in tqdm(image_list):

            ids = image_path.split("/")[-1].replace(".h5", "")

            h5f = h5py.File(image_path, 'r')

            image = h5f['data'][:]
            label = h5f['label'][:]

            # determine patient id from filename
            pid = os.path.basename(image_path).split('_')[0]
            if clinical:
                cv = clinical_map.get(pid, None)
            else:
                cv = None

            prediction, score_map = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes, clinical_vec=cv)

            metric = calculate_metric_percase(prediction == 1, label == 1)

            total_metric[0, :] += metric
            
            f.writelines("{},{},{},{},{}\n".format(ids, metric[0], metric[1], metric[2], metric[3]))

            # --- Load original NCCT for correct spatial metadata ---
            ref_spacing = (1.0, 1.0, 1.0)
            ref_origin = (0.0, 0.0, 0.0)
            ref_direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

            if original_img_dir is not None:
                ref_ncct_path = os.path.join(original_img_dir, pid, 'ses-01',
                                             f'{pid}_ses-01_ncct.nii.gz')
                if os.path.exists(ref_ncct_path):
                    ref_img = sitk.ReadImage(ref_ncct_path)
                    ref_spacing = ref_img.GetSpacing()
                    ref_origin = ref_img.GetOrigin()
                    ref_direction = ref_img.GetDirection()
                else:
                    print(f"  WARNING: reference NCCT not found at {ref_ncct_path}, "
                          f"saving with default (1,1,1) spacing.")

            # --- Transpose from nibabel (i,j,k) to SimpleITK (k,j,i) order ---
            pred_arr = np.transpose(prediction.astype(np.uint8), (2, 1, 0))
            lab_arr = np.transpose(label.astype(np.uint8), (2, 1, 0))
            if image.ndim == 4:
                img_arr = np.transpose(image[-1], (2, 1, 0))  # save last channel (CTA) as representative
            else:
                img_arr = np.transpose(image, (2, 1, 0))

            pred_itk = sitk.GetImageFromArray(pred_arr)
            pred_itk.SetSpacing(ref_spacing)
            pred_itk.SetOrigin(ref_origin)
            pred_itk.SetDirection(ref_direction)
            sitk.WriteImage(pred_itk, test_save_path +
                            "/{}_pred.nii.gz".format(ids))

            img_itk = sitk.GetImageFromArray(img_arr.astype(np.float32))
            img_itk.SetSpacing(ref_spacing)
            img_itk.SetOrigin(ref_origin)
            img_itk.SetDirection(ref_direction)
            sitk.WriteImage(img_itk, test_save_path +
                            "/{}_img.nii.gz".format(ids))

            lab_itk = sitk.GetImageFromArray(lab_arr)
            lab_itk.SetSpacing(ref_spacing)
            lab_itk.SetOrigin(ref_origin)
            lab_itk.SetDirection(ref_direction)
            sitk.WriteImage(lab_itk, test_save_path +
                            "/{}_lab.nii.gz".format(ids))
            
            # Save probability map as .npz
            np.savez_compressed(test_save_path + f"/{ids}.npz", probabilities=score_map.astype(np.float32))
            
        f.writelines("Mean metrics,{},{},{},{}".format(total_metric[0, 0] / len(image_list), total_metric[0, 1] / len(
            image_list), total_metric[0, 2] / len(image_list), total_metric[0, 3] / len(image_list)))
    f.close()
    print("Testing end")
    return total_metric / len(image_list)


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / \
            (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice

def calculate_metric_percase(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        ravd = abs(metric.binary.ravd(pred, gt))
        hd = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        assd = metric.binary.assd(pred, gt)
        return np.array([dice, ravd, hd, asd, assd])
    else:
        return np.array([0, 1, 0, 0, 0]) 