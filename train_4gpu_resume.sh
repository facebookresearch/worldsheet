EXP_NAME=$1
ARGS=$2

mmf_run config=projects/neural_rendering/configs/replica/${EXP_NAME}.yaml \
    datasets=replica \
    model=mesh_renderer \
    env.save_dir=./save/replica/${EXP_NAME} \
    run_type=train_val \
    checkpoint.resume=True ${ARGS}
