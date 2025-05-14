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
"""Module to get the absolute path of a reference dataset for tests

.. moduleauthor:: Denis Arrivault
"""

from __future__ import print_function, division

import os


def get_dataset_path(filename):
    """Return the absolute path of a reference dataset for tests

    - Input parameter:

    :param str filename: File name of the file containing reference data
        for tests (which must be in ``skgilearn/tests/datasets/``)

    - Output parameters:

    :returns: The absolute path where the file with name **filename** is stored
    :rtype: str

    """

    datasets_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(datasets_path, filename)
