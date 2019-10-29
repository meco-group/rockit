#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Optimal control problem."""

from setuptools import setup, find_packages

version = "0.1.5"

setup(
    name='rockit-meco',
    version=version,
    author="MECO-Group",
    author_email="joris.gillis@kuleuven.be",
    description="Rapid Optimal Control Kit",
    license='LICENSE',
    keywords="OCP optimal control casadi",
    url='https://gitlab.mech.kuleuven.be/meco-software/rockit',
    packages=find_packages(exclude=['tests', 'examples']),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'casadi>=3.4,<4.0',
        'numpy>=1.14,<2.0',
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
    download_url='https://gitlab.mech.kuleuven.be/meco-software/rockit/-/archive/v%s/rockit-v%s.tar.gz' % (version, version)
)
