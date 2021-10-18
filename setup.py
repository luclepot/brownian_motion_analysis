import io
import os
import sys
from shutil import rmtree
from setuptools import find_packages, setup, Command
import setuptools
# Package meta-data.
NAME = 'brownian_motion_analysis'
DESCRIPTION = 'Brownian motion analysis code for 111B'
VERSION = '0.1.0'
AUTHOR = 'Luc Le Pottier'
EMAIL = 'luclepot@berkeley.edu'
URL = 'https://github.com/luclepot/brownian_motion_analysis'
REQUIRES_PYTHON = '>=3.6.5'

# What packages are required for this module to be executed?
REQUIRED = [
    'matplotlib',
    'numpy>=1.19.2',
    'pandas>=1.1.4',
    'opencv_python>=4.5.3.56',
    'trackpy==0.5.0',
    'PIMS==0.5'
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name=NAME,
    version=VERSION,
    install_requires=REQUIRED,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    project_urls={
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # package_dir={"": "src"},
    py_modules=["brownian_motion_analysis"],
    # scripts=['modes.py'],
    # packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)