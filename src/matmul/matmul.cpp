/*
*    Copyright (C) 2024 The University of Tokyo
*    
*    File:          /src/matmul/matmul.cpp
*    Project:       pybind_example
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  11-01-2024 21:42:23
*    Last Modified: 11-01-2024 21:42:23
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// check availability of OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

template<typename T>
py::array_t<T> matmul(py::array_t<T> &arr1, py::array_t<T> &arr2) {
	const auto &buf1 = arr1.request();
	const auto &buf2 = arr2.request();
	// check shape
	if (buf1.ndim != 2 || buf2.ndim != 2) {
		throw std::runtime_error("Number of dimensions must be two");
	}
	if (buf1.shape[1] != buf2.shape[0]) {
		throw std::runtime_error("Dimension mismatch");
	}
	T *ptr1 = (T *)buf1.ptr;
	T *ptr2 = (T *)buf2.ptr;
	// declare result array
	py::array_t<T> res = py::array_t<T>(buf1.shape[0] * buf2.shape[1]);
	T *res_ptr = (T *)res.request().ptr;

	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (int i = 0; i < buf1.shape[0]; i++) {
		for (int j = 0; j < buf2.shape[1]; j++) {
			T sum = 0;
			for (int k = 0; k < buf1.shape[1]; k++) {
				sum += ptr1[i * buf1.shape[1] + k] * ptr2[k * buf2.shape[1] + j];
			}
			res_ptr[i * buf2.shape[1] + j] = sum;
		}
	}
	return res;
}

PYBIND11_MODULE(matmul, m) {
	m.def("matmul", &matmul<float>, "matmul");
	m.def("matmul", &matmul<double>, "matmul");
	m.def("matmul", &matmul<int>, "matmul");
	m.def("matmul", &matmul<long>, "matmul");
}
