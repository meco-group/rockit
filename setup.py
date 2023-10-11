#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Optimal control problem."""

from setuptools import setup, find_packages
import glob
import os

version = "0.1.36"

setup(
    name='rockit-meco',
    version=version,
    author="MECO-Group",
    author_email="joris.gillis@kuleuven.be",
    description="Rapid Optimal Control Kit",
    license='LICENSE',
    keywords="OCP optimal control casadi",
    url='https://gitlab.kuleuven.be/meco-software/rockit',
    packages=find_packages(exclude=['tests', 'examples']),
    include_package_data=True,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'casadi>=3.5,<4.0',
        'numpy',
        'matplotlib',
        'scipy'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering'
    ],
    download_url='https://gitlab.kuleuven.be/meco-software/rockit/-/archive/v%s/rockit-v%s.tar.gz' % (version, version)
)
