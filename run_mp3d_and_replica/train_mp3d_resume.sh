# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

EXP_NAME=$1
ARGS=$2

python mmf_cli/run.py config=projects/neural_rendering/configs/synsin_habitat/${EXP_NAME}.yaml \
    datasets=synsin_habitat \
    model=mesh_renderer \
    env.save_dir=./save/synsin_habitat/${EXP_NAME} \
    run_type=train \
    dataset_config.synsin_habitat.multiview_data_dir=${MP3D_SAVE_IMAGE_DIR}/synsin_mp3d \
    dataset_config.synsin_habitat.multiview_image_dir=${MP3D_SAVE_IMAGE_DIR}/synsin_mp3d \
    dataset_config.synsin_habitat.annotations.test="['synsin_habitat/defaults/annotations/imdb_mp3d_train.npy']" \
    checkpoint.reset.all=False checkpoint.resume_file=None checkpoint.resume=True ${ARGS}
