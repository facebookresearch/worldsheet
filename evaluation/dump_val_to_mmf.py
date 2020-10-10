# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import json
import os
import time

import cv2
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader

from evaluation.metrics import perceptual_sim, psnr, ssim_metric

from models.base_model import BaseModel
from models.depth_model import Model
from models.networks.pretrained_networks import PNet
from models.networks.sync_batchnorm import convert_model

from options.options import get_dataset, get_model
from options.test_options import ArgumentParser
from utils.geometry import get_deltas

torch.manual_seed(0)
torch.backends.cudnn.benchmark = True


def _get_pytorch3d_RT_from_P(P):
    P = P.copy()
    # change from Habitat coordinates to PyTorch3D coordinates
    P[0] *= -1  # flip X axis
    P[2] *= -1  # flip Z axis
    R = P[0:3, 0:3].T  # to row major
    T = P[0:3, 3]
    return R, T


def save_batch(batch, batch_idx, result_folder):
    from skimage.io import imsave

    im_dir = os.path.join(result_folder, "images")
    data_dir = os.path.join(result_folder, "data")

    batch_size = len(batch["images"][-1])
    global_idx_begin = batch_size * batch_idx

    imgs_0 = (batch["images"][0] * 0.5 + 0.5).permute(0, 2, 3, 1).numpy()
    imgs_1 = (batch["images"][1] * 0.5 + 0.5).permute(0, 2, 3, 1).numpy()
    depths_0 = batch["depths"][0].permute(0, 2, 3, 1).numpy()
    depths_1 = batch["depths"][1].permute(0, 2, 3, 1).numpy()
    P_0 = batch['cameras'][0]['P'].numpy()
    P_1 = batch['cameras'][1]['P'].numpy()

    vis_mask = None
    if 'vis_mask' in batch:
        vis_mask = batch['vis_mask'].permute(0, 2, 3, 1).numpy()

    for i in range(batch_size):
        n_sample = global_idx_begin + i

        # imsave(os.path.join(im_dir, 'sample_{:08d}_im_{:04d}.png'.format(n_sample, 0)), imgs_0[i])
        # imsave(os.path.join(im_dir, 'sample_{:08d}_im_{:04d}.png'.format(n_sample, 1)), imgs_1[i])

        R_0, T_0 = _get_pytorch3d_RT_from_P(P_0[i])
        R_1, T_1 = _get_pytorch3d_RT_from_P(P_1[i])
        np.savez_compressed(
            os.path.join(data_dir, 'sample_{:08d}.npz'.format(n_sample)),
            rgbs=np.stack([imgs_0[i], imgs_1[i]]),
            depths=np.stack([depths_0[i], depths_1[i]]),
            camera_Rs=np.stack([R_0, R_1]),
            camera_Ts=np.stack([T_0, T_1]),
            vis_mask=vis_mask[i]
        )


def worker_init_fn(worker_id):
    torch.manual_seed(worker_id)


def check_initial_batch(batch, dataset):
    try:
        if dataset == 'replica':
            np.testing.assert_allclose(batch['cameras'][0]['P'].data.numpy().ravel(), 
                np.loadtxt('./data/files/eval_cached_cameras_replica.txt'))
        else:
            np.testing.assert_allclose(batch['cameras'][0]['P'].data.numpy().ravel(), 
                np.loadtxt('./data/files/eval_cached_cameras_mp3d.txt'))
    except Exception as e:
        raise Exception("\n \nThere is an error with your setup or options. \
            \n\nYour results will NOT be comparable with results in the paper or online.")


if __name__ == "__main__":
    print("STARTING MAIN METHOD...", flush=True)
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    test_ops, _ = ArgumentParser().parse()

    # Load model to be tested
    MODEL_PATH = test_ops.old_model
    BATCH_SIZE = test_ops.batch_size

    opts = torch.load(MODEL_PATH)["opts"]
    opts.config = '/private/home/ronghanghu/workspace/habitat-api/configs/tasks/pointnav_rgbd.yaml'
    print("Model is: ", MODEL_PATH)

    opts.image_type = test_ops.image_type
    opts.only_high_res = False

    opts.train_depth = False

    if test_ops.dataset:
        opts.dataset = test_ops.dataset

    Dataset = get_dataset(opts)

    # Update parameters
    opts.render_ids = test_ops.render_ids
    opts.gpu_ids = test_ops.gpu_ids
    opts.jitter_quaternions_angle = test_ops.jitter_quaternions_angle
    opts.images_before_reset = test_ops.images_before_reset

    torch_devices = [int(gpu_id.strip()) for gpu_id in opts.gpu_ids.split(",")]
    device = "cuda:" + str(torch_devices[0])

    # Create dummy depth model for doing sampling images
    sampled_model = Model(opts).to(device)
    sampled_model.eval()

    data = Dataset("val", opts, vectorize=False)
    dataloader = DataLoader(
        data,
        shuffle=False,
        drop_last=False,
        batch_size=BATCH_SIZE,
        num_workers=test_ops.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    iter_data_loader = iter(dataloader)
    next(iter_data_loader)

    N = test_ops.images_before_reset * 18 * BATCH_SIZE

    os.makedirs(os.path.join(test_ops.result_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(test_ops.result_folder, "data"), exist_ok=True)

    # Calculate the metrics and store for each index
    # and change in angle and translation
    for i in tqdm(range(0, N // BATCH_SIZE)):
        with torch.no_grad():
            batch = next(iter_data_loader)
            output_img = (batch["images"][-1] * 0.5 + 0.5).cpu()

            _, new_imgs = sampled_model(batch)
            sampled_img = (new_imgs["SampledImg"] * 0.5 + 0.5).cpu()

        mask = (output_img == sampled_img)
        mask = mask.float().min(dim=1, keepdim=True)[0]
        batch["vis_mask"] = mask

        save_batch(batch, i, test_ops.result_folder)
