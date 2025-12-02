import types
from solver import Solver

# Build a config object matching main.py parser defaults but with quantum disabled
config = types.SimpleNamespace(
    quantum=False,
    patches=1,
    layer=1,
    qubits=8,
    cycle='classical',
    lambda_cycle=0.0,
    qdi_reps=2,
    qdi_layers=1,
    qdi_batch=False,
    complexity='mr',
    g_conv_dim=[128],
    d_conv_dim=[[128,64], 128, [128,64]],
    g_repeat_num=6,
    d_repeat_num=6,
    lambda_cls=1.0,
    lambda_rec=10.0,
    lambda_gp=10.0,
    post_method='softmax',
    batch_size=2,
    num_iters=2,
    num_iters_decay=2500,
    g_lr=0.0001,
    d_lr=0.0001,
    dropout=0.0,
    n_critic=5,
    beta1=0.5,
    beta2=0.999,
    resume_iters=None,
    test_iters=5000,
    num_workers=1,
    mode='train',
    use_tensorboard=False,
    mol_data_dir='data/gdb9.sparsedataset',
    log_dir='qgan-hg-mr-q8-l1/logs',
    model_save_dir='qgan-hg-mr-q8-l1/models',
    sample_dir='qgan-hg-mr-q8-l1/samples',
    result_dir='qgan-hg-mr-q8-l1/results',
    log_step=1,
    sample_step=1000,
    model_save_step=1000,
    lr_update_step=500
)

if __name__ == '__main__':
    solver = Solver(config)
    solver.train()
