EXP_NAME=$1
ARGS=$2

python mmf_cli/run.py config=projects/neural_rendering/configs/synsin_habitat/${EXP_NAME}.yaml \
    datasets=synsin_habitat \
    model=mesh_renderer \
    env.save_dir=./save/synsin_habitat/${EXP_NAME} \
    dataset_config.synsin_habitat.multiview_data_dir=/checkpoint/ronghanghu/neural_rendering_datasets/synsin_replica \
    dataset_config.synsin_habitat.multiview_image_dir=/checkpoint/ronghanghu/neural_rendering_datasets/synsin_replica \
    dataset_config.synsin_habitat.annotations.test="['synsin_habitat/defaults/annotations/imdb_replica_val.npy']" \
    checkpoint.resume=True checkpoint.resume_best=True run_type=test ${ARGS}