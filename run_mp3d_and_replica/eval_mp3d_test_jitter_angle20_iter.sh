# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

EXP_NAME=$1
ITER=$2
ARGS=$3

python mmf_cli/run.py config=projects/neural_rendering/configs/synsin_habitat/${EXP_NAME}.yaml \
    datasets=synsin_habitat \
    model=mesh_renderer \
    env.save_dir=./save/synsin_habitat/${EXP_NAME} \
    dataset_config.synsin_habitat.multiview_data_dir=${MP3D_SAVE_IMAGE_DIR}/synsin_mp3d \
    dataset_config.synsin_habitat.multiview_image_dir=${MP3D_SAVE_IMAGE_DIR}/synsin_mp3d \
    dataset_config.synsin_habitat.annotations.test="['synsin_habitat/defaults/annotations/imdb_mp3d_test_jitter_angle20.npy']" \
    checkpoint.resume_file=save/synsin_habitat/${EXP_NAME}/models/model_${ITER}.ckpt run_type=test ${ARGS}
echo "exp:" ${EXP_NAME}