import numpy as np
import torch
import os
import matplotlib.cm as cm
import torch.nn as nn
import cv2
from PIL import Image
import accimage
import torchvision
import torchvision.transforms as transforms
from skimage import color
import torch.nn.functional as F

def _recover_rgb_image(img):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
    img_rec = img * std[:, None, None] + mean[:, None, None]
    img_rec = (img_rec * 255).clamp(0, 255).type(torch.uint8).cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray(img_rec)


def _to_2d_map(arr, name):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected {name} to be 2D, got shape {arr.shape}")
    return arr


def _colorize_map(map_2d, cmap_name='jet'):
    map_2d = map_2d.astype(np.float32)
    max_val = float(map_2d.max())
    if max_val > 0:
        map_2d = map_2d / max_val
    else:
        map_2d = np.zeros_like(map_2d, dtype=np.float32)
    color_map = getattr(cm, cmap_name)(map_2d)[:, :, :3]
    return Image.fromarray((color_map * 255).astype(np.uint8))


def vis_rgb_gt_amp(img_paths, imgs, img_masks, anomaly_maps, method, root_out, dataset_name):
    if imgs.shape[-1] != img_masks.shape[-1]:
        imgs = F.interpolate(imgs, size=img_masks.shape[-1], mode='bilinear', align_corners=False)

    for idx, (img_path, img, img_mask, anomaly_map) in enumerate(zip(img_paths, imgs, img_masks, anomaly_maps)):
        parts = img_path.split('/')
        needed_parts = parts[1:-1]
        specific_root = '/'.join(needed_parts)
        img_num = parts[-1].split('.')[0]

        out_dir = f'{root_out}/{method}/{specific_root}'
        os.makedirs(out_dir, exist_ok=True)
        img_path = f'{out_dir}/{img_num}_img.png'
        img_ano_path = f'{out_dir}/{img_num}_amp.png'
        mask_path = f'{out_dir}/{img_num}_mask.png'

        # RGB image
        img_rec = _recover_rgb_image(img)
        img_rec.save(img_path)

        # anomaly map -> [H, W]
        anomaly_map = _to_2d_map(anomaly_map, "anomaly_map")
        anomaly_map = _colorize_map(anomaly_map, cmap_name='jet')

        img_rec_anomaly_map = Image.blend(img_rec, anomaly_map, alpha=0.4)
        img_rec_anomaly_map.save(img_ano_path)

        # mask -> [H, W]
        img_mask = _to_2d_map(img_mask, "img_mask")
        img_mask = (img_mask * 255).astype(np.uint8)
        img_mask = np.repeat(img_mask[:, :, None], 3, axis=2)
        img_mask = Image.fromarray(img_mask)
        img_mask.save(mask_path)


def vis_conf_map(img_paths, imgs, conf_maps, method, root_out, dataset_name, suffix='mconf'):
    if conf_maps.ndim == 4:
        conf_maps = conf_maps[:, 0]

    for img_path, img, conf_map in zip(img_paths, imgs, conf_maps):
        parts = img_path.split('/')
        needed_parts = parts[1:-1]
        specific_root = '/'.join(needed_parts)
        img_num = parts[-1].split('.')[0]

        out_dir = f'{root_out}/{method}/{specific_root}'
        os.makedirs(out_dir, exist_ok=True)
        conf_path = f'{out_dir}/{img_num}_{suffix}.png'
        conf_overlay_path = f'{out_dir}/{img_num}_{suffix}_overlay.png'

        img_rec = _recover_rgb_image(img)
        conf_map = _to_2d_map(conf_map, "conf_map")
        conf_heatmap = _colorize_map(conf_map, cmap_name='viridis')
        conf_heatmap.save(conf_path)

        conf_overlay = Image.blend(img_rec, conf_heatmap, alpha=0.45)
        conf_overlay.save(conf_overlay_path)
