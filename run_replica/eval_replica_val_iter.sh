EXP_NAME=$1
ITER=$2
ARGS=$3

python mmf_cli/run.py config=projects/neural_rendering/configs/replica/${EXP_NAME}.yaml \
    datasets=replica \
    model=mesh_renderer \
    env.save_dir=./save/replica/${EXP_NAME} \
    checkpoint.resume_file=save/replica/${EXP_NAME}/models/model_${ITER}.ckpt run_type=val ${ARGS}
echo "exp:" ${EXP_NAME}