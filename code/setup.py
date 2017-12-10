from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'tracing',
  ext_modules = cythonize("tracing.pyx"),
  include_dirs=[numpy.get_include()]
)