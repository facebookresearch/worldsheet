EXP_NAME=$1
ARGS=$2

mmf_run config=projects/neural_rendering/configs/replica/${EXP_NAME}.yaml \
    datasets=replica \
    model=mesh_renderer \
    env.save_dir=./save/replica/${EXP_NAME} \
    model_config.mesh_renderer.save_forward_results=True \
    model_config.mesh_renderer.forward_results_dir=save/visualization/${EXP_NAME}/train \
    checkpoint.resume=True checkpoint.resume_best=True run_type=val ${ARGS}

mmf_run config=projects/neural_rendering/configs/replica/${EXP_NAME}.yaml \
    datasets=replica \
    model=mesh_renderer \
    env.save_dir=./save/replica/${EXP_NAME} \
    model_config.mesh_renderer.save_forward_results=True \
    model_config.mesh_renderer.forward_results_dir=save/visualization/${EXP_NAME}/test \
    checkpoint.resume=True checkpoint.resume_best=True run_type=test ${ARGS}
