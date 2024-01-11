pybind11_add_module(matmul matmul.cpp)

if (CUDA_FOUND)
	pybind11_add_module(matmul_cuda matmul_cuda.cu)
endif()

if (OpenCL_FOUND)
	pybind11_add_module(matmul_opencl matmul_opencl.cpp)
	target_include_directories(matmul_opencl PRIVATE ${OpenCL_INCLUDE_DIRS})
	target_link_libraries(matmul_opencl PRIVATE ${OpenCL_LIBRARIES})
	target_compile_options(matmul_opencl PRIVATE
		$<$<CONFIG:Debug>:-DVERBOSE>
	)
endif()