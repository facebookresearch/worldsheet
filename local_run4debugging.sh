EXP_NAME=$1
ARGS=$2

python mmf_cli/run.py config=projects/neural_rendering/configs/replica/${EXP_NAME}.yaml \
    datasets=replica \
    model=mesh_renderer \
    env.save_dir=/tmp/mmf_nr_debug/replica/${EXP_NAME} \
    training.batch_size=8 \
    run_type=train_val ${ARGS}
