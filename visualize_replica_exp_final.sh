EXP_NAME=$1
ARGS=$2

python mmf_cli/run.py config=projects/neural_rendering/configs/replica/${EXP_NAME}.yaml \
    datasets=replica \
    model=mesh_renderer \
    env.save_dir=./save/replica/${EXP_NAME} \
    model_config.mesh_renderer.save_forward_results=True \
    model_config.mesh_renderer.forward_results_dir=save/visualization/${EXP_NAME}__final/train \
    checkpoint.resume=True run_type=val ${ARGS}

python mmf_cli/run.py config=projects/neural_rendering/configs/replica/${EXP_NAME}.yaml \
    datasets=replica \
    model=mesh_renderer \
    env.save_dir=./save/replica/${EXP_NAME} \
    model_config.mesh_renderer.save_forward_results=True \
    model_config.mesh_renderer.forward_results_dir=save/visualization/${EXP_NAME}__final/test \
    checkpoint.resume=True run_type=test ${ARGS}
