#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Optimal control problem."""

from setuptools import setup, find_packages

setup(
    name='ocpx',
    version="0.1.0",
    author="MECO-Group",
    author_email="joris.gillis@kuleuven.be",
    description="Optimal control problem helper library",
    license="Unknown",
    keywords="OCP Casadi",
    packages=find_packages(exclude=['tests', 'examples']),
    install_requires=[
        'casadi>=3.4,<4.0',
        'numpy>=1.14,<2.0',
    ],
)
