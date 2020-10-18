EXP_NAME=$1
ARGS=$2

python mmf_cli/run.py config=projects/neural_rendering/configs/replica/${EXP_NAME}.yaml \
    datasets=replica \
    model=mesh_renderer \
    env.save_dir=./save/replica/${EXP_NAME} \
    model_config.mesh_renderer.save_forward_results=True \
    model_config.mesh_renderer.forward_results_dir=save/visualization/${EXP_NAME}/largebl_train \
    dataset_config.replica.annotations.val="['replica/defaults/annotations/imdb_mini_large_baseline_train.npy']" \
    dataset_config.replica.annotations.test="['replica/defaults/annotations/imdb_mini_large_baseline_test.npy']" \
    checkpoint.resume_file=None checkpoint.resume=True checkpoint.resume_best=True run_type=val ${ARGS}

python mmf_cli/run.py config=projects/neural_rendering/configs/replica/${EXP_NAME}.yaml \
    datasets=replica \
    model=mesh_renderer \
    env.save_dir=./save/replica/${EXP_NAME} \
    model_config.mesh_renderer.save_forward_results=True \
    model_config.mesh_renderer.forward_results_dir=save/visualization/${EXP_NAME}/largebl_test \
    dataset_config.replica.annotations.val="['replica/defaults/annotations/imdb_mini_large_baseline_train.npy']" \
    dataset_config.replica.annotations.test="['replica/defaults/annotations/imdb_mini_large_baseline_test.npy']" \
    checkpoint.resume_file=None checkpoint.resume=True checkpoint.resume_best=True run_type=test ${ARGS}
