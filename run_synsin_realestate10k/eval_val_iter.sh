EXP_NAME=$1
ARGS=$2

python mmf_cli/run.py config=projects/neural_rendering/configs/synsin_realestate10k/${EXP_NAME}.yaml \
    datasets=synsin_realestate10k \
    model=mesh_renderer \
    env.save_dir=./save/synsin_realestate10k/${EXP_NAME} \
    checkpoint.resume_file=save/synsin_realestate10k/${EXP_NAME}/models/model_${ITER}.ckpt run_type=test ${ARGS}
echo "exp:" ${EXP_NAME}