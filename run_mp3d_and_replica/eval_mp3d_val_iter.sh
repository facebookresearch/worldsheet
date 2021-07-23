EXP_NAME=$1
ITER=$2
ARGS=$3

python mmf_cli/run.py config=projects/neural_rendering/configs/synsin_habitat/${EXP_NAME}.yaml \
    datasets=synsin_habitat \
    model=mesh_renderer \
    env.save_dir=./save/synsin_habitat/${EXP_NAME} \
    dataset_config.synsin_habitat.multiview_data_dir=${MP3D_SAVE_IMAGE_DIR}/synsin_mp3d \
    dataset_config.synsin_habitat.multiview_image_dir=${MP3D_SAVE_IMAGE_DIR}/synsin_mp3d \
    dataset_config.synsin_habitat.annotations.test="['synsin_habitat/defaults/annotations/imdb_mp3d_val.npy']" \
    checkpoint.resume_file=save/synsin_habitat/${EXP_NAME}/models/model_${ITER}.ckpt run_type=val ${ARGS}
echo "exp:" ${EXP_NAME}