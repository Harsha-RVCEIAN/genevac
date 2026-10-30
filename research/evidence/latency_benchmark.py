"""
Empirical Latency Benchmark on Target Machine
Measures actual runtime on CPU for:
1. Pure Kinematic LinearTrajectoryPredictor (current baseline)
2. Representative Lightweight PyTorch GRU Model (candidate learned model)
Across batch sizes: 10, 50, 100, 250 agents.
"""

import time
import numpy as np
# pyrefly: ignore[missing-import]
import torch
# pyrefly: ignore[missing-import]
import torch.nn as nn
from genevac.prediction import LinearTrajectoryPredictor


class SimpleGRUPredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, output_steps=30):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_steps * 2)
        self.output_steps = output_steps

    def forward(self, x):
        # x: (B, T_obs, 2)
        _, h_n = self.gru(x)
        # h_n[-1]: (B, hidden_dim)
        out = self.fc(h_n[-1])
        return out.view(-1, self.output_steps, 2)


def benchmark_linear_predictor(batch_sizes=[10, 50, 100, 250], history_len=20, horizon=30, runs=100):
    predictor = LinearTrajectoryPredictor()
    results = {}

    for B in batch_sizes:
        # Create synthetic history for B agents
        batch_histories = [
            [(float(t) * 0.1, float(t) * 0.05) for t in range(history_len)]
            for _ in range(B)
        ]

        # Warm-up
        for _ in range(10):
            for h in batch_histories:
                _ = predictor.predict(h, horizon=horizon)

        # Timed runs
        start = time.perf_counter()
        for _ in range(runs):
            for h in batch_histories:
                _ = predictor.predict(h, horizon=horizon)
        elapsed = (time.perf_counter() - start) / runs

        results[B] = {
            "batch_time_ms": elapsed * 1000.0,
            "per_agent_us": (elapsed / B) * 1_000_000.0
        }
    return results


def benchmark_gru_predictor(batch_sizes=[10, 50, 100, 250], history_len=20, horizon=30, runs=100):
    device = torch.device("cpu")
    model = SimpleGRUPredictor(input_dim=2, hidden_dim=64, num_layers=2, output_steps=horizon).to(device)
    model.eval()

    results = {}
    with torch.no_grad():
        for B in batch_sizes:
            x = torch.randn(B, history_len, 2, dtype=torch.float32, device=device)

            # Warm-up
            for _ in range(10):
                _ = model(x)

            # Timed runs
            start = time.perf_counter()
            for _ in range(runs):
                _ = model(x)
            elapsed = (time.perf_counter() - start) / runs

            results[B] = {
                "batch_time_ms": elapsed * 1000.0,
                "per_agent_us": (elapsed / B) * 1_000_000.0
            }
    return results


def main():
    print("=" * 60)
    print("EMPIRICAL LATENCY BENCHMARK (PyTorch 2.4.1+cpu / Windows AMD64)")
    print("=" * 60)

    batch_sizes = [10, 50, 100, 250]
    history_len = 20
    horizon = 30
    runs = 200

    print(f"Parameters: History={history_len} steps, Horizon={horizon} steps, Repetitions={runs}")
    print("\n1. LinearTrajectoryPredictor (Kinematic Constant Velocity)")
    lin_res = benchmark_linear_predictor(batch_sizes, history_len, horizon, runs)
    for B, m in lin_res.items():
        print(f"  Batch {B:3d} agents: Total = {m['batch_time_ms']:.4f} ms | Per-agent = {m['per_agent_us']:.2f} µs")

    print("\n2. Representative PyTorch 2-layer GRU (Lightweight Neural Predictor, CPU)")
    gru_res = benchmark_gru_predictor(batch_sizes, history_len, horizon, runs)
    for B, m in gru_res.items():
        print(f"  Batch {B:3d} agents: Total = {m['batch_time_ms']:.4f} ms | Per-agent = {m['per_agent_us']:.2f} µs")

    # Measure tensor conversion and preprocessing overhead
    print("\n3. Preprocessing / Tensor Conversion Overhead (List of Tuples -> FloatTensor)")
    for B in batch_sizes:
        raw_data = [[(float(t), float(t)) for t in range(history_len)] for _ in range(B)]
        start = time.perf_counter()
        for _ in range(runs):
            arr = np.array(raw_data, dtype=np.float32)
            t_tensor = torch.from_numpy(arr)
        overhead_ms = ((time.perf_counter() - start) / runs) * 1000.0
        print(f"  Batch {B:3d} agents: Conversion = {overhead_ms:.4f} ms")


if __name__ == "__main__":
    main()

