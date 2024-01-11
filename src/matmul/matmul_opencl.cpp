/*
*    Copyright (C) 2024 The University of Tokyo
*    
*    File:          /src/matmul/matmul_opencl.cpp
*    Project:       pybind_example
*    Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
*    Created Date:  11-01-2024 21:37:16
*    Last Modified: 11-01-2024 21:37:21
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <string>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#ifdef VERBOSE
#define debug_print(fmt, ...) \
			do { fprintf(stderr, fmt, ##__VA_ARGS__); } while (0)
#define debug(ctx) \
			do { ctx } while (0)
#else
#define debug_print(fmt, ...) \
			do { } while (0)
#define debug(ctx) \
			do { } while (0)
#endif

namespace py = pybind11;
using namespace std;

// kernel program
template<typename T>
struct KernelSource
{
	static const char *get_source();
};

template<>
const char *KernelSource<float>::get_source()
{
	return R"(
	__kernel void matmul_kernel(const int width,
						 __global const float* a,
						 __global const float* b,
						 __global float* c) {
		int row = get_global_id(1);
		int col = get_global_id(0);
		float value = 0.0f;
		for (int i = 0; i < width; i++)
			value += a[row * width + i] * b[i * width + col];
		c[row * width + col] = value;
	}
)";
}

template<>
const char *KernelSource<double>::get_source()
{
	return R"(
	__kernel void matmul_kernel(const int width,
						 __global const double* a,
						 __global const double* b,
						 __global double* c) {
		int row = get_global_id(1);
		int col = get_global_id(0);
		double value = 0.0;
		for (int i = 0; i < width; i++)
			value += a[row * width + i] * b[i * width + col];
		c[row * width + col] = value;
	}
)";
}

template<>
const char *KernelSource<int>::get_source()
{
	return R"(
	__kernel void matmul_kernel(const int width,
						 __global const int* a,
						 __global const int* b,
						 __global int* c) {
		int row = get_global_id(1);
		int col = get_global_id(0);
		int value = 0;
		for (int i = 0; i < width; i++)
			value += a[row * width + i] * b[i * width + col];
		c[row * width + col] = value;
	}
)";
}

cl_context context;
cl_command_queue command_queue;

runtime_error error_msg_with_code(string msg, cl_int err)
{
	return std::move(runtime_error(msg + " (code: " + to_string(err) + ")"));
}

cl_platform_id get_target_platform()
{
	cl_int err;
	cl_uint platformCount;
	debug_print("Getting platform...\n");
	// Get the number of platforms
	err = clGetPlatformIDs(0, nullptr, &platformCount);
	if (err != CL_SUCCESS) {
		throw error_msg_with_code("Error: Failed to get platform count", err);
	}
	debug_print("platformCount: %d\n", platformCount);
	vector<cl_platform_id> platforms(platformCount);
	err = clGetPlatformIDs(platformCount, platforms.data(), nullptr);
	if (err != CL_SUCCESS) {
		throw error_msg_with_code("Error: Failed to get platform IDs", err);
	}

	cl_platform_id platform;
	// get environment variables
	const char *env = getenv("CL_PLATFORM");
	if (env != nullptr) {
		debug_print("Target platform is specified by environment variable CL_PLATFORM %s\n", env);
		int select_id = stoi(env);
		if (select_id >= platformCount) {
			throw runtime_error("Error: Invalid platform ID");
		}
		platform = platforms[select_id];
	} else {
		platform = platforms[0];
	}

	// get platform info
	size_t size;
	err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &size);
	string name(size, '\0');
	err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, size, const_cast<char*>(name.data()), nullptr);
	debug_print("Platform: %s\n", name.c_str());
	return platform;
}

cl_device_id get_target_device(cl_platform_id platform)
{
	cl_int err;
	cl_device_id target_device;
	cl_uint deviceCount;
	debug_print("Getting device...\n");
	// get device count
	vector<cl_device_id> devices(deviceCount);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &deviceCount);
	if (err != CL_SUCCESS) {
		throw error_msg_with_code("Error: Failed to get device count", err);
	}
	debug_print("deviceCount: %d\n", deviceCount);

	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr);
	if (err != CL_SUCCESS) {
		throw error_msg_with_code("Error: Failed to get device IDs", err);
	}
	// get environment variables
	const char *env = getenv("CL_DEVICE");
	if (env != nullptr) {
		debug_print("Target device is specified by environment variable CL_DEVICE %s\n", env);
		int select_id = stoi(env);
		if (select_id >= deviceCount) {
			throw runtime_error("Error: Invalid device ID");
		}
		target_device = devices[select_id];
	} else {
		target_device = devices[0];
	}
	// get device name
	debug(
		size_t size;
		err = clGetDeviceInfo(target_device, CL_DEVICE_NAME, 0, nullptr, &size);
		string device_name(size, '\0');
		err = clGetDeviceInfo(target_device, CL_DEVICE_NAME, size, const_cast<char*>(device_name.data()), nullptr);
		debug_print("Device: %s\n", device_name.c_str());
	);
	return target_device;
}

template<typename T>
cl_kernel cl_init()
{
	cl_int err;
	cl_platform_id platform_id = get_target_platform();
	cl_device_id device_id = get_target_device(platform_id);
	// create context
	debug_print("Creating context...\n");
	context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		throw error_msg_with_code("Error: Failed to create a compute context", err);
	}
	// create command queue
	command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);

	// load program
	debug_print("Loading program...\n");
	const char *kernelSource = KernelSource<T>::get_source();
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);
	if (err != CL_SUCCESS) {
		throw error_msg_with_code("Error: Failed to create compute program", err);
	}
	// build program
	debug_print("Building program...\n");
	err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (err != CL_SUCCESS) {
		throw error_msg_with_code("Error: Failed to build program executable", err);
	}
	// create kernel
	debug_print("Creating kernel...\n");
	cl_kernel kernel = clCreateKernel(program, "matmul_kernel", &err);
	if (err != CL_SUCCESS) {
		throw error_msg_with_code("Error: Failed to create compute kernel", err);
	}
	clFinish(command_queue);
	return kernel;

}
template<typename T>
py::array_t<T> matmul(py::array_t<T> &array_a, py::array_t<T> &array_b) {

	cl_int err;

	const auto &buf_info_a = array_a.request();
	const auto &buf_info_b = array_b.request();
	// check shape
	if (buf_info_a.ndim != 2 || buf_info_b.ndim != 2) {
		throw error_msg_with_code("Number of dimensions must be two", err);
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
		throw error_msg_with_code("Dimension mismatch", err);
	}

	// host pointer
	T *a_host = (T *)buf_info_a.ptr;
	T *b_host = (T *)buf_info_b.ptr;
	// declare result array
	py::array_t<T> res = py::array_t<T>(a_h * b_w);
	T *c_host = (T *)res.request().ptr;

	// init
	cl_kernel kernel = cl_init<T>();

	// device pointer
	debug_print("Allocating device memory...\n");
	cl_mem a_device, b_device, c_device;
	a_device = clCreateBuffer(context, CL_MEM_READ_ONLY, a_w * a_h * sizeof(T), NULL, &err);
	b_device = clCreateBuffer(context, CL_MEM_READ_ONLY, b_w * b_h * sizeof(T), NULL, &err);
	c_device = clCreateBuffer(context, CL_MEM_READ_WRITE, c_w * c_h * sizeof(T), NULL, &err);
	if (err != CL_SUCCESS) {
		throw error_msg_with_code("Error: Failed to allocate device memory", err);
	}

	// data copy
	debug_print("Copying data from host to device...\n");
	err = clEnqueueWriteBuffer(command_queue, a_device, CL_TRUE, 0, a_w * a_h * sizeof(T), a_host, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(command_queue, b_device, CL_TRUE, 0, b_w * b_h * sizeof(T), b_host, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		throw error_msg_with_code("Error: Failed to copy data from host to device", err);
	}

	// launch kernel
	debug_print("Launching kernel...\n");
	clSetKernelArg(kernel, 0, sizeof(int), &a_w); 
	clSetKernelArg(kernel, 1, sizeof(a_device), &a_device);
	clSetKernelArg(kernel, 2, sizeof(b_device), &b_device);
	clSetKernelArg(kernel, 3, sizeof(c_device), &c_device);
	size_t global[2] = { (size_t)c_w, (size_t)c_h };
	err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, NULL, 0, NULL, NULL);
	clFinish(command_queue);
	if (err != CL_SUCCESS) {
		throw error_msg_with_code("Error: Failed to execute kernel", err);
	}

	// copy result
	debug_print("Copying data from device to host...\n");
	err = clEnqueueReadBuffer(command_queue, c_device, CL_TRUE, 0, c_w * c_h * sizeof(T), c_host, 0, NULL, NULL);

	// release
	debug_print("Releasing resources...\n");
	clReleaseMemObject(a_device);
	clReleaseMemObject(b_device);
	clReleaseMemObject(c_device);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	return res;
}

PYBIND11_MODULE(matmul_opencl, m)
{
	m.def("matmul", &matmul<float>);
	m.def("matmul", &matmul<double>);
	m.def("matmul", &matmul<int>);
}