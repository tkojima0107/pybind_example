###
#   Copyright (C) 2024 The University of Tokyo
#   
#   File:          /setup.py
#   Project:       pybind_example
#   Author:        Takuya Kojima in The University of Tokyo (tkojima@hal.ipc.i.u-tokyo.ac.jp)
#   Created Date:  11-01-2024 21:40:07
#   Last Modified: 11-01-2024 21:40:14
###


from setuptools import setup, find_packages, Extension
import subprocess
import os


__version__ = '1.0.0'

try:
    import pybind11
except ImportError:
    ext_modules = []
else:
    from setuptools.command.build_ext import build_ext
    class CMakeExtension(Extension):
        def __init__(self, name, cmake_src_dir):
            super().__init__(name, sources=[])
            self.cmake_src_dir = os.path.abspath(cmake_src_dir)


    class CMakeBuild(build_ext):
        def run(self):
            # check if build directory exists
            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)

            for ext in self.extensions:
                abs_build_lib = os.path.abspath(self.build_lib)
                cmake_args = [
                    f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={abs_build_lib}{os.sep}{ext.name}'
                ]
                build_dir = f"{self.build_temp}{os.sep}{ext.name}"
                os.makedirs(build_dir, exist_ok=True)
                # print pwd with subprocess
                subprocess.check_call(['pwd'], cwd=build_dir)

                # configure
                subprocess.check_call(['cmake', ext.cmake_src_dir] + cmake_args, cwd=build_dir)
                # build
                subprocess.check_call(['cmake', '--build', '.'], cwd=build_dir)

    ext_modules = [
        CMakeExtension("pybind_example", 'src')
    ]


setup(
    name='pybind-example',
    version=__version__,
    license='proprietary',
    description='Example of packaged C++ extension with pybind11',

    author='Takuya Kojima',
    author_email='tkojima@hal.ipc.i.u-tokyo.ac.jp',
    url='https://www.tkojima.me',

    cmdclass={"build_ext": CMakeBuild},

    ext_modules=ext_modules,

)