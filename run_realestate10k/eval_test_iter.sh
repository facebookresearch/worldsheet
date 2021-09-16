# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

EXP_NAME=$1
ITER=$2
ARGS=$3

python mmf_cli/run.py config=projects/neural_rendering/configs/synsin_realestate10k/${EXP_NAME}.yaml \
    datasets=synsin_realestate10k \
    model=mesh_renderer \
    env.save_dir=./save/synsin_realestate10k/${EXP_NAME} \
    checkpoint.resume_file=save/synsin_realestate10k/${EXP_NAME}/models/model_${ITER}.ckpt run_type=test ${ARGS}
echo "exp:" ${EXP_NAME}