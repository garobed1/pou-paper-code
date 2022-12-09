from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("pou_cython_ext.pyx")
)