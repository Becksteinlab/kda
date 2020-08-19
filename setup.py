# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
#
# Kinetic Diagram Analysis
#
# Released under the GNU Public Licence, v3 or any higher version

import sys
from setuptools import setup, find_packages
import versioneer

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

with open("README.md", "r") as handle:
    long_description = handle.read()

setup(
    # Self-descriptive entries which should always be present
    name='kda',
    author='Nikolaus Awtrey',
    author_email='nawtrey@asu.edu',
    url='https://github.com/Becksteinlab/kda/',
    description="Kinetic Diagram Analysis tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='LGPLv3',
    keywords="science kinetics chemistry physics",
    project_urls={
        'Source': 'https://github.com/Becksteinlab/kda/',
        'Issue Tracker': 'https://github.com/Becksteinlab/kda/issues',
    },
    packages=find_packages(),
    install_requires=[
        'numpy',
        'networkx',
        'scipy',
        'sympy',
        'matplotlib',
    ],
    tests_require=[
        'pytest',
        'pytest-xdist'
    ],
    include_package_data=True,
    setup_requires=[] + pytest_runner,
    )
