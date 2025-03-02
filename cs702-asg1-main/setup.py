# setup.py

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension(
    "q3",
    ["q3_simulator.py"],
    include_dirs=[numpy.get_include()]
)

setup(
    name="cs702_asg1",
    ext_modules=cythonize([ext]),
    version="0.1",
    zip_safe=False,
)
