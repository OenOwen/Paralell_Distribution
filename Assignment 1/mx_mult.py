import os
import numpy as np
import time

# Turn off multithreading
os.environ["OMP_NUM_THREADS"] = "1"

times = []
for i in range(10):
    A = np.random.rand(1024, 1024)
    B = np.random.rand(1024, 1024)
    
    start_time = time.time()
    np.dot(A, B)
    end_time = time.time()
    
    times.append(end_time - start_time)

average_time = sum(times) / 10

print("Average time for numpy: ", average_time)