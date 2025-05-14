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

from __future__ import division, print_function
import numpy as np
import unittest
from splearn.datasets.base import load_data_sample
from splearn.datasets.base import _read_dimension, _load_file_doublelecture
from splearn.tests.datasets.get_dataset_path import get_dataset_path


class UnitaryTest(unittest.TestCase):

    def test_load_data_sample(self):
        adr = get_dataset_path("3.pautomac.train")
        # adr = get_dataset_path("essai")

        l = load_data_sample(adr=adr)
        nbL = l.nbL
        nbEx = l.nbEx
        sample = l.data
        self.assertEqual(nbL, 4)
        self.assertEqual(nbEx, 20000)
        nbEx2 = sample.shape[0]
        self.assertEqual(nbEx, nbEx2)
        lmax = sample.shape[1]
        self.assertEqual(67, lmax)

    def test_read_dimension(self):
        adr = get_dataset_path("3.pautomac.train")
        # adr = get_dataset_path("essai")

        l = _read_dimension(adr=adr)
        nbEx = l[0]
        max_length = l[1]
        self.assertEqual(nbEx, 20000)
        self.assertEqual(max_length, 67)


    def test_load_file_doublelecture(self):
        adr = get_dataset_path("3.pautomac.train")
        l = _load_file_doublelecture(adr=adr)
        # adr = get_dataset_path("essai")
        nbL = l[0]
        nbEx = l[1]
        sample = l[2]
        self.assertEqual(nbL, 4)
        self.assertEqual(nbEx, 20000)
        nbEx2 = sample.shape[0]
        self.assertEqual(nbEx, nbEx2)
        lmax = sample.shape[1]
        self.assertEqual(67, lmax)


if __name__ == '__main__':
    unittest.main()
