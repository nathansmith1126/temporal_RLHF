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
from splearn.datasets.data_sample import DataSample, SplearnArray
from splearn.tests.datasets.get_dataset_path import get_dataset_path
from splearn.spectral import Spectral


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


    def test_assignement(self):

        adr = get_dataset_path("essai")
        # adr = get_dataset_path("essai")

        s = load_data_sample(adr=adr)
        cl = Spectral()
        cl.polulate_dictionnaries(s.data)
        self.assertEqual(s.nbL,s.data.nbL)
        self.assertEqual(s.nbEx, s.data.nbEx)
        with self.assertRaises(TypeError):
            s.nbL= 2.0
        with self.assertRaises(ValueError):
            s.nbL = -1
        # nbEx assignment
        with self.assertRaises(TypeError):
            s.nbEx = 6.0

        with self.assertRaises(ValueError):
            s.nbEx = -6

    def test_load_Splearn_array(self):
        adr = get_dataset_path("3.pautomac.train")
        # adr = get_dataset_path("essai")

        data = load_data_sample(adr=adr)
        cl = Spectral(partial=False)
        cl.polulate_dictionnaries(data.data)
        nbL = data.data.nbL
        nbEx = data.data.nbEx
        sample = data.data .sample
        pref = data.data.pref
        suff = data.data.suff

        self.assertEqual(nbL, 4)
        self.assertEqual(nbEx, 20000)
        nbEx2 = sum([sample[w] for w in sample])
        self.assertEqual(nbEx, nbEx2)
        nbPref1 = sum([sample[w]*(len(w)+1) for w in sample])
        nbPref2 = sum([pref[w] for w in pref])
        self.assertEqual(nbPref1, nbPref2)
        nbSuff1 = sum([sample[w]*(len(w)+1) for w in sample])
        nbSuff2 = sum([suff[w] for w in suff])
        self.assertEqual(nbSuff1, nbSuff2)

        cl = Spectral(version = 'factor', partial=False)
        cl.polulate_dictionnaries(data.data)
        fact = data.data.fact
        nbFact1 = sum([sample[w]*(len(w)+1)*(len(w)+2)/2 for w in sample])
        nbFact2 = sum([fact[w] for w in fact])
        self.assertEqual(nbFact1, nbFact2)

    def test_select_columns(self):
        adr = get_dataset_path("0.spice.train")
        pT = load_data_sample(adr=adr)
        cl = Spectral(partial=False)
        cl.polulate_dictionnaries(pT.data)
        # lR = pT.data.select_rows(nb_rows_max = 10, version = 'classic')
        # lC = pT.data.select_columns(nb_columns_max = 10, version = 'classic')
        # self.assertEqual(lR, [(), (3,), (3, 0), (3, 3), (3, 0, 3), (3, 1),
        #                       (3, 3, 0), (3, 0, 3, 0), (3, 1, 3),
        #                       (3, 3, 1)])
        # self.assertEqual(lC, [(), (3,), (0,), (3, 3), (3, 0), (0, 3), (2,),
        #                   (1,), (1, 3), (3, 0, 3)])

        cl = Spectral(version = 'prefix', partial=False)
        cl.polulate_dictionnaries(pT.data)
        # lRp = pT.data.select_rows(nb_rows_max = 10, version = 'prefix')
        # lCp = pT.data.select_columns(nb_columns_max = 10, version = 'prefix')
        # self.assertEqual(lRp, [(), (3,), (3, 0), (3, 0, 0), (3, 0, 1),
        #                        (3, 0, 2), (3, 0, 3), (3, 0, 0, 0),
        #                        (3, 0, 0, 1), (3, 0, 0, 2)])
        # self.assertEqual(lCp, [(), (3,), (0,), (1,), (3, 0), (3, 3), (2,),
        #                    (0, 3), (1, 3), (3, 1)])

        cl = Spectral(version = 'factor', partial=False)
        cl.polulate_dictionnaries(pT.data)
        # lRf = pT.data.select_rows(nb_rows_max = 10, version = 'factor')
        # lCf = pT.data.select_columns(nb_columns_max = 10, version = 'factor')
        # self.assertEqual(lRf,  [(), (3,), (0,), (1,), (3, 0), (3, 3), (2,),
        #                         (3, 1), (0, 3), (1, 3)])
        # self.assertEqual(lCf,  [(), (3,), (0,), (1,), (2,), (3, 0), (3, 3),
        #                         (1, 3), (0, 3), (3, 1)])


if __name__ == '__main__':
    unittest.main()
