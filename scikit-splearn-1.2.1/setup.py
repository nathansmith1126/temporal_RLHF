#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
#
# Copyright(c) 2016-2018
# -----------------
#
# * LabEx Archimède: http://labex-archimede.univ-amu.fr/
# * Laboratoire d'Informatique et Systèmes : http://www.lis-lab.fr/
#
# Contributors:
# ------------
#
# * François Denis <francois.denis_AT_lis-lab.fr>
# * Rémi Eyraud <remi.eyraud_AT_lis-lab.fr>
# * Denis Arrivault <contact.dev_AT_lis-lab.fr>
# * Dominique Benielli <dominique.benielli_AT_univ-amu.fr>
#
# Description:
# -----------
#
# scitkit-splearn is a toolbox in
# python for spectral learning algorithms.
#
# Version:
# -------
#
# * splearn version = 1.2.1
#
# Licence:
# -------
#
# License: 3-clause BSD
#
#
# ######### COPYRIGHT #########

from __future__ import print_function
import os
import sys
import shutil

# Always prefer setuptools over distutils
try:
    from setuptools import setup
    from setuptools.command.clean import clean
    from setuptools.command.sdist import sdist
except ImportError:
    from distutils.core import setup
    from distutils.command.clean import clean
    from distutils.command.sdist import sdist

USE_COPYRIGHT = True
try:
    from copyright import writeStamp, eraseStamp
except ImportError:
    USE_COPYRIGHT = False


###################
# Get scikit-splearn version
####################
def get_version():
    v_text = open('VERSION').read().strip()
    v_text_formted = '{"' + v_text.replace('\n', '","').replace(':', '":"')
    v_text_formted += '"}'
    v_dict = eval(v_text_formted)
    return v_dict["splearn"]


########################
# Set ltfat __version__
########################
def set_version(splearn_dir, version):
    filename = os.path.join(splearn_dir, '__init__.py')
    buf = ""
    for line in open(filename, "rb"):
        if not line.decode("utf8").startswith("__version__ ="):
            buf += line.decode("utf8")
    f = open(filename, "wb")
    f.write(buf.encode("utf8"))
    f.write(('__version__ = "%s"\n' % version).encode("utf8"))


######################
# Custom clean command
######################
class m_clean(clean):
    """ Remove build directories, and compiled file in the source tree"""

    def run(self):
        clean.run(self)
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('splearn'):
            for filename in filenames:
                if (filename.endswith('.so') or
                        filename.endswith('.pyd') or
                        filename.endswith('.dll') or
                        filename.endswith('.pyc')):
                    os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))


##############################
# Custom sdist command
##############################
class m_sdist(sdist):
    """ Build source package

    WARNING : The stamping must be done on an default utf8 machine !
    """

    def run(self):
        if USE_COPYRIGHT:
            writeStamp()
            sdist.run(self)
            # eraseStamp()
        else:
            sdist.run(self)


##########################
# File path read command
##########################
def read(*paths):
    """Build a file path from *paths* and return the contents."""
    with open(os.path.join(*paths), 'r', encoding='utf-8') as f:
        return f.read()


####################
# Setup method
####################
def setup_package():
    """ Setup function"""
    # set version
    VERSION = get_version()
    splearn_dir = 'splearn'
    set_version(splearn_dir, VERSION)

    setup(name="scikit-splearn",
          version=VERSION,
          description='Python module for spectral learning of weighted automata',
          long_description=(read('README.rst') + '\n\n' +
                            read('HISTORY.rst') + '\n\n' +
                            read('AUTHORS.rst')),
          packages=["splearn", "splearn.datasets", "splearn.tests", "splearn.tests.datasets"],
          package_data={'splearn.tests.datasets': ['*']},
          url="https://gitlab.lis-lab.fr/dev/scikit-splearn.git",
          license='new BSD',
          author='François Denis and Rémi Eyraud and Denis Arrivault and Dominique Benielli',
          author_email='francois.denis@lis-lab.fr, ' +
                       'remi.eyraud@lis-lab.fr, ' +
                       'denis.arrivault@lis-lab.fr, ' +
                       'dominique.benielli@univ-amu.fr ',
          cmdclass={'clean': m_clean, 'sdist': m_sdist},
          classifiers=['Development Status :: 5 - Production/Stable',
                       'Intended Audience :: Science/Research',
                       'Intended Audience :: Developers',
                       'Natural Language :: English',
                       'License :: OSI Approved :: BSD License',
                       'Operating System :: MacOS :: MacOS X',
                       'Operating System :: POSIX :: Linux',
                       'Programming Language :: Python :: 3.4',
                       'Topic :: Scientific/Engineering',
                       'Topic :: Scientific/Engineering :: Mathematics'
                       ],
          install_requires=['numpy>=1.8', 'scipy>=0.16', 'six>=1.10', 'scikit-learn>=0.17.1'],
          )

if __name__ == "__main__":
    setup_package()
