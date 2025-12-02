import time
import torch
from quantum_layers import QDILayer


def benchmark(batch_size=8, input_dim=64, n_qubits=8, reps=1, layers=1):
    x = torch.randn(batch_size, input_dim)

    qdi_un = QDILayer(input_dim=input_dim, n_qubits=n_qubits, n_reps=reps, n_layers=layers, batch=False)
    qdi_b = QDILayer(input_dim=input_dim, n_qubits=n_qubits, n_reps=reps, n_layers=layers, batch=True)

    # sync weights
    with torch.no_grad():
        qdi_b.weights.copy_(qdi_un.weights)
        qdi_b.encoder.weight.copy_(qdi_un.encoder.weight)
        qdi_b.encoder.bias.copy_(qdi_un.encoder.bias)

    # warmup
    for _ in range(2):
        _ = qdi_un(x)
        _ = qdi_b(x)

    t0 = time.time()
    _ = qdi_un(x)
    t_un = time.time() - t0

    t0 = time.time()
    _ = qdi_b(x)
    t_b = time.time() - t0

    print(f"batch={batch_size} unbatched_time={t_un:.4f}s batched_time={t_b:.4f}s speedup={t_un/t_b if t_b>0 else float('inf'):.2f}")


if __name__ == '__main__':
    benchmark(batch_size=4, input_dim=32, n_qubits=4)
    benchmark(batch_size=8, input_dim=64, n_qubits=6)
    benchmark(batch_size=16, input_dim=128, n_qubits=8)
