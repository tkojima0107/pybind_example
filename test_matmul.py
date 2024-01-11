import numpy as np
import time

A_N = (3 << 9)
A_M = (4 << 9)
B_N = A_M
B_M = (4 << 9)

dtype = np.int32

# create two random matrices
a = np.random.rand(A_N, A_M).astype(dtype)
b = np.random.rand(B_N, B_M).astype(dtype)

# compute the matrix product
print("--- running on numpy.matmul ---")
start = time.time()
answer = np.matmul(a, b)
end = time.time()
print(f"elapsed time: {end - start} sec\n")

from pybind_example import matmul

print("--- running on C++ implementation ---")
start = time.time()
c_cpu = matmul.matmul(a, b).reshape((A_N, B_M))
end = time.time()
print("correct" if np.allclose(answer, c_cpu, atol=1e-10) else "incorrect")
print(f"elapsed time: {end - start} sec\n")

try:
    from pybind_example import matmul_cuda
except ImportError:
    print("CUDA implementation is not available")
else:
    print("--- running on CUDA implementation ---")
    start = time.time()
    c_cuda = matmul_cuda.matmul(a, b).reshape((A_N, B_M))
    end = time.time()
    print("correct" if np.allclose(answer, c_cuda, atol=1e-10) else "incorrect")
    print(f"elapsed time: {end - start} sec\n")

try:
    from pybind_example import matmul_opencl
except ImportError:
    print("OpenCL implementation is not available")
else:
    print("--- running on OpenCL implementation ---")
    start = time.time()
    c_opencl = matmul_opencl.matmul(a, b).reshape((A_N, B_M))
    end = time.time()
    print("correct" if np.allclose(answer, c_opencl, atol=1e-10) else "incorrect")
    print(f"elapsed time: {end - start} sec")



