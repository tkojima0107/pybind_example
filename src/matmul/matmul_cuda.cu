#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cuda_runtime.h>


namespace py = pybind11;

#define BLOCL_SIZE 16

template <typename T>
__global__ void matmul_kernel(T* a, T* b, T* c, long a_width, long c_width, long c_height)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	T sum = 0;

	if (idx < c_width && idy < c_height)
	{
		T sum = 0;
		for (int k = 0; k < a_width; k++)
		{
			sum += a[idy * a_width + k] * b[k * c_width + idx];
		}
		c[idy * c_width + idx] = sum;
	}
}


template<typename T>
py::array_t<T> matmul(py::array_t<T> &array_a, py::array_t<T> &array_b) {
	const auto &buf_info_a = array_a.request();
	const auto &buf_info_b = array_b.request();
	// check shape
	if (buf_info_a.ndim != 2 || buf_info_b.ndim != 2) {
		throw std::runtime_error("Number of dimensions must be two");
	}
	// set size
	const int a_w = buf_info_a.shape[1];
	const int a_h = buf_info_a.shape[0];
	const int b_w = buf_info_b.shape[1];
	const int b_h = buf_info_b.shape[0];
	const int c_w = b_w;
	const int c_h = a_h;
	// check dimension
	if (a_w != b_h) {
		throw std::runtime_error("Dimension mismatch");
	}

	// host pointer
	T *a_host = (T *)buf_info_a.ptr;
	T *b_host = (T *)buf_info_b.ptr;
	// declare result array
	py::array_t<T> res = py::array_t<T>(a_h * b_w);
	T *c_host = (T *)res.request().ptr;

	// device pointer
	T *a_device, *b_device, *c_device;

	// malloc
	cudaError_t error = cudaMalloc((void**)&a_device, a_w * a_h * sizeof(T));
	if (error != cudaSuccess) {
		throw std::runtime_error(cudaGetErrorString(error));
	}
	error = cudaMalloc((void**)&b_device, b_w * b_h * sizeof(T));
	if (error != cudaSuccess) {
		throw std::runtime_error(cudaGetErrorString(error));
	}
	error = cudaMalloc((void**)&c_device, b_w * a_h * sizeof(T));
	if (error != cudaSuccess) {
		throw std::runtime_error(cudaGetErrorString(error));
	}

	// mempcy
	error = cudaMemcpy(a_device, a_host, a_w * a_h * sizeof(T), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		throw std::runtime_error(cudaGetErrorString(error));
	}
	error = cudaMemcpy(b_device, b_host, b_w * b_h * sizeof(T), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		throw std::runtime_error(cudaGetErrorString(error));
	}

	// // thread params
	dim3 dimBlock(BLOCL_SIZE, BLOCL_SIZE);
	dim3 dimGrid((c_w + dimBlock.x - 1) / dimBlock.x, (c_h + dimBlock.y - 1) / dimBlock.y);
	
	// run kernel
	matmul_kernel<T><<<dimGrid, dimBlock>>>(a_device, b_device, c_device, a_w, c_w, c_h);

	// flush
	std::flush(std::cout);

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		throw std::runtime_error(cudaGetErrorString(error));
	}
	cudaDeviceSynchronize();
	error = cudaMemcpy(c_host, c_device, c_w * c_h * sizeof(T), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		throw std::runtime_error(cudaGetErrorString(error));
	}

	return res;
}

PYBIND11_MODULE(matmul_cuda, m)
{
	m.def("matmul", &matmul<float>);
	m.def("matmul", &matmul<double>);
	m.def("matmul", &matmul<int>);
	m.def("matmul", &matmul<long>);
}