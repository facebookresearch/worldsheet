EXP_NAME=$1
ARGS=$2

python mmf_cli/run.py config=projects/neural_rendering/configs/diode/${EXP_NAME}.yaml \
    datasets=diode \
    model=mesh_renderer \
    env.save_dir=./save/diode/${EXP_NAME} \
    run_type=train_val \
    checkpoint.reset.all=False checkpoint.resume_file=None checkpoint.resume=True ${ARGS}
