EXP_NAME=$1
ARGS=$2

python mmf_cli/run.py config=projects/neural_rendering/configs/diode/${EXP_NAME}.yaml \
    datasets=diode \
    model=mesh_renderer \
    env.save_dir=./save/diode/${EXP_NAME} \
    model_config.mesh_renderer.save_forward_results=True \
    model_config.mesh_renderer.forward_results_dir=save/visualization_diode/${EXP_NAME}/val \
    checkpoint.reset.all=True checkpoint.resume=True checkpoint.resume_best=True run_type=val ${ARGS}

python mmf_cli/run.py config=projects/neural_rendering/configs/diode/${EXP_NAME}.yaml \
    datasets=diode \
    model=mesh_renderer \
    env.save_dir=./save/diode/${EXP_NAME} \
    model_config.mesh_renderer.save_forward_results=True \
    model_config.mesh_renderer.forward_results_dir=save/visualization_diode/${EXP_NAME}/train \
    checkpoint.reset.all=True checkpoint.resume=True checkpoint.resume_best=True run_type=test ${ARGS}