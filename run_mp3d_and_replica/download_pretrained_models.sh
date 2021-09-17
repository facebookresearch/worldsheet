# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

mkdir -p ./save/synsin_habitat/mp3d_nodepth_perceptual_l1laplacian/models/
wget -O \
    ./save/synsin_habitat/mp3d_nodepth_perceptual_l1laplacian/models/model_40000.ckpt \
    https://dl.fbaipublicfiles.com/worldsheet/pretrained_models/synsin_habitat/mp3d_nodepth_perceptual_l1laplacian/models/model_40000.ckpt

mkdir -p ./save/synsin_habitat/mp3d_nodepth_perceptual_l1laplacian_inpaintGonly_freezemesh/models/
wget -O \
    ./save/synsin_habitat/mp3d_nodepth_perceptual_l1laplacian_inpaintGonly_freezemesh/models/model_40000.ckpt \
    https://dl.fbaipublicfiles.com/worldsheet/pretrained_models//synsin_habitat/mp3d_nodepth_perceptual_l1laplacian_inpaintGonly_freezemesh/models/model_40000.ckpt
