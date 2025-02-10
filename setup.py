#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Optimal control problem."""

from setuptools import setup, find_packages
import glob
import os

version = "0.2.8"

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):

        for filename in filenames:
            if filename.endswith(".pyd"): continue
            full = os.path.join(path, filename)
            if "__pycache__" in full: continue
            if "acados_matlab_octave" in full: continue
            if "examples" in full: continue
            if "/." in full: continue
            if "/test/" in full: continue
            if "/docs/" in full: continue
            if "/external/utils" in full: continue
            if "/external/external/" in full and ("hpipm" not in full and "blasfeo" not in full and "/external/external/CMakeLists.txt" not in full): continue
            paths.append(full)
    return paths

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
        'scipy',
        'future-fstrings'
    ],
    data_files=[('acados',package_files('rockit/external/acados/external')),
                ('acados_interface',package_files('rockit/external/acados/interface_generation'))],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering'
    ],
    download_url='https://gitlab.kuleuven.be/meco-software/rockit/-/archive/v%s/rockit-v%s.tar.gz' % (version, version)
)
