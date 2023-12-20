from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
	Extension(
		"fastmixture.em",
		["fastmixture/em.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	), 
	Extension(
		"fastmixture.em_batch",
		["fastmixture/em_batch.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	),
	Extension(
		"fastmixture.svd",
		["fastmixture/svd.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	),
	Extension(
		"fastmixture.shared",
		["fastmixture/shared.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	)
]

setup(
	name="fastmixture",
	version="0.3",
	description="Fast Ancestry Estimation",
	author="Jonas Meisner",
	packages=["fastmixture"],
	entry_points={
		"console_scripts": ["fastmixture=fastmixture.main:main"]
	},
	python_requires=">=3.6",
	install_requires=[
		"cython",
		"numpy"
	],
	ext_modules=cythonize(extensions),
	include_dirs=[numpy.get_include()]
)
