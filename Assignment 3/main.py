import cupy as cp
import time
import numpy as np

# Matrix size
n = 1024
num_runs = 10

def create_mx(n):
    np.random.seed(int(time.time()))  # like srand(time(0))
    mx = np.random.randint(1, 101, size=n*n, dtype=np.int32)
    return mx.reshape(n, n)

# Create random matrices on CPU, then transfer to GPU
A = cp.asarray(create_mx(n))
B = cp.asarray(create_mx(n))

# Warm-up to initialize GPU context
cp.matmul(cp.ones((2, 2), dtype=cp.int32), cp.ones((2, 2), dtype=cp.int32))

# Timing multiple runs
times = []

for i in range(num_runs):
    start_time = time.time()
    C = cp.matmul(A, B)
    cp.cuda.Stream.null.synchronize()  # ensure GPU work is done
    elapsed = time.time() - start_time
    times.append(elapsed)

# Calculate average
avg_time = sum(times) / num_runs

# Output formatted like C++ cout
print(f"\nTime taken for 3loop Matrix multiplication (CUDA): {avg_time * 1000:.2f} milliseconds ({avg_time:.4f} seconds)")
