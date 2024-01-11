# pybind_example

This repository is an example of how to use pybind11 with OpenMP and CUDA.

## Pre-requisites
* CMake >= 3.20
* pybind11 >= 2.6
* python3-devel

### Optional
* CUDA
* OpenCL library & OpenCL headers

## steps to build and install

1. clone the repository

2. install pybind11
```
pip3 install pybind11
```

3. build and install this repository
```
pip3 install .
```

To tell CMake where to find the OpenCL library, you can set the environment variables `OPENCL_LIBRARY`.
