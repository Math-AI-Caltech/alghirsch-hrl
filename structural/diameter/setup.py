from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(
    name        = "diameter",
    packages    = find_packages(),
    ext_modules = [
        cpp_extension.CppExtension(
            "diameter._C", ["diameter/src/floyd_warshall.cpp"],
            extra_compile_args = {"cxx": ["-DPy_LIMITED_API=0x03090000","-fopenmp"]}, # min version: python3.9
            py_limited_api = True)],
    cmdclass    = {"build_ext": cpp_extension.BuildExtension},
    options     = {"bdist_wheel": {"py_limited_api": "cp39"}},
    install_requires = ["torch"])
