#
#    Copyright (C) 2024 The University of Tokyo
#    
#    File:          /Downloads/pybind_example/src/CMakeLists.txt
#    Project:       pybind_example
#    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#    Created Date:  11-01-2024 21:39:57
#    Last Modified: 11-01-2024 22:49:32
#


cmake_minimum_required(VERSION 3.20)

project(pybind_kernel VERSION 0.1.0)
set(Python_VIRTUALENV FIRST)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# find the pybind11 package for giving hints
execute_process(
  COMMAND
	${Python3_EXECUTABLE}  -c "import os; import pybind11; print(os.path.dirname(pybind11.__file__))"
  OUTPUT_VARIABLE
    PYBIND11_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED HINTS ${PYBIND11_SITE_PACKAGES})

# find OpenMP support for MacOS
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
	execute_process(COMMAND brew --prefix libomp
					OUTPUT_VARIABLE OpenMP_ROOT
					OUTPUT_STRIP_TRAILING_WHITESPACE)  

	set(OpenMP_C_LIB_NAMES "omp")
	set(OpenMP_CXX_LIB_NAMES "omp")
	set(OpenMP_omp_LIBRARY "${OpenMP_ROOT}/lib/")

	include_directories("${OpenMP_ROOT}/include")
	link_directories("${OpenMP_ROOT}/lib/")
	set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -lomp")

endif()

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
else()
	message(STATUS "Skip building CUDA libraries")
endif()

# if env OpenCL_LIBRARY is set, use it
if (DEFINED ENV{OpenCL_LIBRARY})
	set(OpenCL_LIBRARY $ENV{OpenCL_LIBRARY})
	message(STATUS "OpenCL_LIBRARY is set to $ENV{OpenCL_LIBRARY}")
endif()
find_package(OpenCL)
if (OpenCL_FOUND)
	message(STATUS "OpenCL found")
else()
	message(STATUS "Skip building OpenCL libraries")
endif()

add_subdirectory(matmul)