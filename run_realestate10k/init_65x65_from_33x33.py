# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os

import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", type=str, required=True, help="input weights of 33x33 mesh"
)
parser.add_argument(
    "--output", type=str, required=True, help="output weights of 65x65 mesh"
)
args = parser.parse_args()

stride8_weights_file = args.input
stride4_weights_file = args.output
stride8_weights = torch.load(stride8_weights_file)


def upgrade_w(w):
    return w.permute(1, 0, 2, 3).repeat(1, 1, 2, 2)


# initialize the grid offset and depth prediction weights
# from the 33x33 mesh by repeating its conv kernels
model = stride8_weights["model"]
for k in [
    "offset_and_depth_predictor.xy_offset_predictor.weight",
    "offset_and_depth_predictor.z_grid_predictor.weight",
]:
    model[k] = upgrade_w(model[k])

# remove redundant buffers of 33x33 mesh size (the model will rebuild them)
for k in [
    "novel_view_projector.unscaled_grid",
    "loss_mesh_laplacian.laplacian",
    "loss_z_grid_l1.sampling_grid",
]:
    model.pop(k)

stride4_weights = stride8_weights
os.makedirs(os.path.dirname(stride4_weights_file), exist_ok=True)
torch.save(stride4_weights, stride4_weights_file)
print(f"new weights for 65x65 mesh saved to {stride4_weights_file}")
