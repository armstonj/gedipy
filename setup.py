#!/usr/bin/env python

"""
Install script for gedipy
"""
from __future__ import print_function

import os
import sys
from numpy.distutils.core import setup, Extension

# don't build extensions if we are in readthedocs
withExtensions = os.getenv('READTHEDOCS', default='False') != 'True'

import gedipy

# use the latest numpy API
NUMPY_MACROS = ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')

# Are we installing the command line scripts?
# This is an experimental option for users who are
# using the Python entry point feature of setuptools and Conda instead
NO_INSTALL_CMDLINE = int(os.getenv('GEDIPY_NOCMDLINE', '0')) > 0
if NO_INSTALL_CMDLINE:
    scriptList = None
else:
    scriptList = ['bin/gedipy_export', 'bin/gedipy_subset', 'bin/gedipy_extract', 'bin/gedipy_grid']

setup(name='gedipy',
      version=gedipy.GEDIPY_VERSION,
      description='Python tools for simplified processing of NASAs Global Ecosystem Dynamics Investigation (GEDI) H5 data products.',
      packages=['gedipy'],
      scripts=scriptList,
      license='LICENSE.txt', 
      url='http://github.com/armstonj/gedipy',
      classifiers=['Intended Audience :: Developers',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8'])
      
