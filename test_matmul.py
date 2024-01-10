import numpy as np
import time

A_N = (3 << 8)
A_M = (4 << 8)
B_N = A_M
B_M = (4 << 8)

dtype = np.int32

# create two random matrices
a = np.random.rand(A_N, A_M).astype(dtype)
b = np.random.rand(B_N, B_M).astype(dtype)

# compute the matrix product
print("--- running on numpy.matmul ---")
start = time.time()
answer = np.matmul(a, b)
end = time.time()
print(f"elapsed time: {end - start} sec")

from pybind_example import matmul, matmul_cuda

print("--- running on C++ implementation ---")
start = time.time()
c_cpu = matmul.matmul(a, b).reshape((A_N, B_M))
end = time.time()
print("correct" if np.allclose(answer, c_cpu, atol=1e-10) else "incorrect")
print(f"elapsed time: {end - start} sec")

print("--- running on CUDA implementation ---")
start = time.time()
c_cuda = matmul_cuda.matmul(a, b).reshape((A_N, B_M))
end = time.time()
print("correct" if np.allclose(answer, c_cuda, atol=1e-10) else "incorrect")
print(f"elapsed time: {end - start} sec")



