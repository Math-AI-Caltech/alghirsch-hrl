import glob
from setuptools import setup, find_packages
from torch.utils import cpp_extension
from torch.utils.cpp_extension import CUDA_HOME

torch_cpp_extension = cpp_extension.CppExtension if CUDA_HOME is None else cpp_extension.CUDAExtension

setup(
    name        = "syzygies",
    packages    = find_packages(),
    ext_modules = [
        torch_cpp_extension(
            "syzygies._C", glob.glob("syzygies/src/*.cpp") + (glob.glob("syzygies/src/*.cu") if CUDA_HOME is not None else []),
            extra_compile_args = {
                "cxx": ["-O3", "-fdiagnostics-color=always", "-DPy_LIMITED_API=0x03090000"] +  # min version: python3.9
                    (["-DBUILD_SGRAPH_CU_"] if CUDA_HOME is not None else []),
                "nvcc": ["-O3"]
                },
            py_limited_api = True)],
    cmdclass    = {"build_ext": cpp_extension.BuildExtension},
    options     = {"bdist_wheel": {"py_limited_api": "cp39"}},
    install_requires = ["torch"])
