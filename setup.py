from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
	Extension(
		"src.svd",
		["src/svd.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	), Extension(
		"src.em",
		["src/em.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	), Extension(
		"src.em_batch",
		["src/em_batch.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	)
]

setup(
	name="fastmixture",
	version="0.2",
	description="Fast Ancestry Estimation",
	author="Jonas Meisner",
	packages=["src"],
	entry_points={
		"console_scripts": ["fastmixture=src.main:main"]
	},
	python_requires=">=3.6",
	install_requires=[
		"cython",
		"numpy"
	],
	ext_modules=cythonize(extensions),
	include_dirs=[numpy.get_include()]
)
