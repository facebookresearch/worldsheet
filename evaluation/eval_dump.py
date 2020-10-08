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

    opts.images_before_reset = test_ops.images_before_reset

    torch_devices = [int(gpu_id.strip()) for gpu_id in opts.gpu_ids.split(",")]
    device = "cuda:" + str(torch_devices[0])

    # Create dummy depth model for doing sampling images
    sampled_model = Model(opts).to(device)
    sampled_model.eval()

    data = Dataset("test", opts, vectorize=False)
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

    if not os.path.exists(test_ops.result_folder):
        os.makedirs(test_ops.result_folder)

    # Calculate the metrics and store for each index
    # and change in angle and translation
    results_all = {}
    for i in tqdm(range(0, N // BATCH_SIZE)):
        with torch.no_grad():
            batch = next(iter_data_loader)
            output_img = (batch["images"][-1] * 0.5 + 0.5).cpu()

            _, new_imgs = sampled_model(batch)
            sampled_img = (new_imgs["SampledImg"] * 0.5 + 0.5).cpu()

        # Check to make sure options were set right and this matches the setup
        # we used, so that numbers are comparable.
        if i == 0:
            check_initial_batch(batch, test_ops.dataset)

        mask = (output_img == sampled_img)
        mask = mask.float().min(dim=1, keepdim=True)[0]
        batch["vis_mask"] = mask
        torch.save(
            batch, os.path.join(test_ops.result_folder, f"batch_{i:08d}.pkl"),
        )
