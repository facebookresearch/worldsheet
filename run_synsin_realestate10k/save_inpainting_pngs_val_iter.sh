EXP_NAME=$1
ITER=$2
ARGS=$3
SUFFIX=$4

python mmf_cli/run.py config=projects/neural_rendering/configs/synsin_realestate10k/${EXP_NAME}.yaml \
    datasets=synsin_realestate10k \
    model=mesh_renderer \
    env.save_dir=./save/synsin_realestate10k/${EXP_NAME} \
    model_config.mesh_renderer.save_forward_results=True model_config.mesh_renderer.save_for_external_inpainting=True \
    model_config.mesh_renderer.forward_results_dir=save/external_inpainting_synsin_realestate10k/${EXP_NAME}/${ITER}/realestate10k_inpainting_val${SUFFIX} \
    checkpoint.resume_file=save/synsin_realestate10k/${EXP_NAME}/models/model_${ITER}.ckpt run_type=val ${ARGS}
