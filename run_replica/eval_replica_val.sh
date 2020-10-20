EXP_NAME=$1
ARGS=$2

python mmf_cli/run.py config=projects/neural_rendering/configs/replica/${EXP_NAME}.yaml \
    datasets=replica \
    model=mesh_renderer \
    env.save_dir=./save/replica/${EXP_NAME} \
    checkpoint.resume_file=None checkpoint.resume=True checkpoint.resume_best=True run_type=val ${ARGS}
echo "exp:" ${EXP_NAME}