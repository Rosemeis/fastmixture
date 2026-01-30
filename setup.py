from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="fastmixture.em",
        sources=["fastmixture/em.pyx"],
        extra_compile_args=["-fopenmp", "-Ofast", "-march=native"],
        extra_link_args=["-fopenmp", "-lm"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        name="fastmixture.svd",
        sources=["fastmixture/svd.pyx"],
        extra_compile_args=["-fopenmp", "-Ofast", "-march=native"],
        extra_link_args=["-fopenmp", "-lm"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        name="fastmixture.shared",
        sources=["fastmixture/shared.pyx"],
        extra_compile_args=["-fopenmp", "-Ofast", "-march=native"],
        extra_link_args=["-fopenmp", "-lm"],
        include_dirs=[numpy.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        },
    ),
)
