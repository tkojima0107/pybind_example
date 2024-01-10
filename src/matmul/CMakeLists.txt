pybind11_add_module(matmul matmul.cpp)
if (CUDA_FOUND)
	pybind11_add_module(matmul_cuda matmul_cuda.cu)
endif()