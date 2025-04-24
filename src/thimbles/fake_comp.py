import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from itertools import product
import time
import os

def fake_computation(x, y):
    time.sleep(0.1)  # Simulate workload
    return x * y + os.getpid()  # Include process ID to observe parallelism

def test_parallel_execution():
    x_vals = np.linspace(100, 100, 100)
    y_vals = np.linspace(100, 100, 100)
    grid = list(product(x_vals, y_vals))

    print(f"Running test on {len(grid)} parameter pairs using joblib...")

    results = Parallel(n_jobs=-1)(
        delayed(fake_computation)(x, y) for (x, y) in tqdm(grid, desc="Parallel Test")
    )

    results = np.array(results).reshape(len(x_vals), len(y_vals))
    print("Computation completed. Result shape:", results.shape)
    print("Sample result:\n", results)

    # Count number of unique process IDs
    pid_list = Parallel(n_jobs=-1)(
        delayed(lambda: os.getpid())() for _ in range(10)
    )
    print("Unique PIDs involved in parallel runs:", set(pid_list))

if __name__ == "__main__":
    test_parallel_execution()
