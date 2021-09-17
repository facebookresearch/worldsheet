# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

mkdir -p ./save/synsin_realestate10k/realestate10k_dscale2_lowerL1_200/models/
wget -O \
    ./save/synsin_realestate10k/realestate10k_dscale2_lowerL1_200/models/model_50000.ckpt \
    https://dl.fbaipublicfiles.com/worldsheet/pretrained_models/synsin_realestate10k/realestate10k_dscale2_lowerL1_200/models/model_50000.ckpt

mkdir -p ./save/synsin_realestate10k/realestate10k_dscale2_stride4ft_lowerL1_200/models/
wget -O \
    ./save/synsin_realestate10k/realestate10k_dscale2_stride4ft_lowerL1_200/models/model_50000.ckpt \
    https://dl.fbaipublicfiles.com/worldsheet/pretrained_models/synsin_realestate10k/realestate10k_dscale2_stride4ft_lowerL1_200/models/model_50000.ckpt
