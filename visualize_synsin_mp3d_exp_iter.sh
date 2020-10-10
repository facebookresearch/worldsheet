EXP_NAME=$1
ITER=$2
ARGS=$3

python mmf_cli/run.py config=projects/neural_rendering/configs/synsin_habitat/${EXP_NAME}.yaml \
    datasets=synsin_habitat \
    model=mesh_renderer \
    env.save_dir=./save/synsin_habitat/${EXP_NAME} \
    model_config.mesh_renderer.save_forward_results=True \
    model_config.mesh_renderer.forward_results_dir=save/visualization_synsin_habitat/${EXP_NAME}/${ITER}/mp3d_test \
    checkpoint.resume_file=save/synsin_habitat/${EXP_NAME}/models/model_${ITER}.ckpt run_type=test ${ARGS}
