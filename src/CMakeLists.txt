cmake_minimum_required(VERSION 3.20)

project(pybind_kernel VERSION 0.1.0)
set(Python_VIRTUALENV FIRST)
find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
message(STATUS "Python3 sitearch: ${Python3_SITEARCH}")
find_package(pybind11 REQUIRED HINTS ${Python3_SITEARCH}/pybind11)
find_package(OpenMP)
find_package(CUDA)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(OpenMP_FOUND)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


if (CUDA_FOUND)
	# automatically add the CUDA bin directory to the path if it is not already included
	set(ENV_PATH "$ENV{PATH}")
	string(FIND "${ENV_PATH}" "${CUDA_TOOLKIT_ROOT_DIR}/bin" cuda_bin_pos)
	if(cuda_bin_pos EQUAL -1)
		set(ENV{PATH} "${CUDA_TOOLKIT_ROOT_DIR}/bin:${ENV_PATH}")
	endif()
	enable_language(CUDA)
endif()

add_subdirectory(matmul)