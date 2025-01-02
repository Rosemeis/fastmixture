from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
	Extension(
		name="fastmixture.em",
		sources=["fastmixture/em.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	),
	Extension(
		name="fastmixture.svd",
		sources=["fastmixture/svd.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	),
	Extension(
		name="fastmixture.shared",
		sources=["fastmixture/shared.pyx"],
		extra_compile_args=['-fopenmp', '-O3', '-g0', '-Wno-unreachable-code'],
		extra_link_args=['-fopenmp'],
		include_dirs=[numpy.get_include()],
		define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
	)
]

setup(
	name="fastmixture",
	version="0.94.3",
	author="Jonas Meisner",
	author_email="meisnerucph@gmail.com",
	description="Fast Ancestry Estimation",
	long_description_content_type="text/markdown",
	long_description=open("README.md").read(),
	url="https://github.com/Rosemeis/fastmixture",
	classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
    ],
	ext_modules=cythonize(extensions),
	python_requires=">=3.10",
	install_requires=[
		"cython>3.0.0",
		"numpy>2.0.0"
	],
	packages=["fastmixture"],
	entry_points={
		"console_scripts": ["fastmixture=fastmixture.main:main"]
	},
)
