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
import unittest
import numpy as np
import scipy.sparse as sps


from splearn.datasets.base import load_data_sample
from splearn.automaton import Automaton
from splearn.spectral import Spectral
from splearn.hankel import Hankel
from splearn.tests.datasets.get_dataset_path import get_dataset_path

class HankelTest(unittest.TestCase):

    def test_CreateHankel_classic(self):
        adr = get_dataset_path("essai")
        # adr = get_dataset_path("3.pautomac.train")
        data = load_data_sample(adr=adr)
        cl = Spectral(partial=False)
        cl.polulate_dictionnaries(data.data)
        lprefix = [()]
        lprefix = lprefix + [(i,) for i in range(data.data.nbL)]
        lprefix = lprefix+[(i, j) for i in range(data.data.nbL)
                           for j in range(data.data.nbL)]
        lsuffix = [()]
        lsuffix = lsuffix + [(i,) for i in range(data.data.nbL)]
        lsuffix = lsuffix + [(i, j) for i in range(data.data.nbL)
                             for j in range(data.data.nbL)]

        for s in data.data.pref.keys():
            self.assertTrue( s in lprefix  )
        for s in data.data.suff.keys():
            self.assertTrue( s in lsuffix )
        l_h = Hankel(sample_instance=data.data, lrows=1, lcolumns=1,
            version="classic", partial=False, sparse=False).lhankel
        self.assertEqual(3, l_h.__len__())
        h_0 = np.array([[ 1.,  1.,  1.,  2.,  1.],
              [ 1.,  0.,  2.,  0.,  0.],
              [ 1.,  0.,  1.,  0.,  0.],
              [ 2.,  0.,  0.,  0.,  0.],
              [ 1.,  0.,  0.,  0.,  0.]])

        h_1 = np.array([[ 1.,  0.,  2.,  0.,  0.],
              [ 0.,  0.,  0.,  0.,  0.],
              [ 0.,  0.,  0.,  0.,  0.],
              [ 0.,  0.,  0.,  0.,  0.],
              [ 0.,  0.,  0.,  0.,  0.]])

        h_2 = np.array([[ 1.,  0.,  1.,  0.,  0.],
              [ 2.,  0.,  0.,  0.,  0.],
              [ 1.,  0.,  0.,  0.,  0.],
              [ 0.,  0.,  0.,  0.,  0.],
              [ 0.,  0.,  0.,  0.,  0.]])

        np.testing.assert_equal(l_h[0], h_0)
        np.testing.assert_equal(l_h[1], h_1)
        np.testing.assert_equal(l_h[2], h_2)

        l_h = Hankel(sample_instance=data.data, lrows=1, lcolumns=1,
                     version="classic", partial=False, sparse=True).lhankel
        self.assertEqual(3, l_h.__len__())
        h_0 = sps.dok_matrix(np.array([[1., 1., 1., 2., 1.],
                        [1., 0., 2., 0., 0.],
                        [1., 0., 1., 0., 0.],
                        [2., 0., 0., 0., 0.],
                        [1., 0., 0., 0., 0.]]))

        h_1 = sps.dok_matrix(np.array([[1., 0., 2., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.]]))

        h_2 = sps.dok_matrix(np.array([[1., 0., 1., 0., 0.],
                        [2., 0., 0., 0., 0.],
                        [1., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.]]))

        np.testing.assert_equal(l_h[0], h_0)
        np.testing.assert_equal(l_h[1], h_1)
        np.testing.assert_equal(l_h[2], h_2)


    def test_CreateHankel_prefix(self):
        adr = get_dataset_path("essai")
        # adr = get_dataset_path("3.pautomac.train")
        data = load_data_sample(adr=adr)
        cl = Spectral(partial=False, version="prefix")
        cl.polulate_dictionnaries(data.data)
        lprefix = [()]
        lprefix = lprefix + [(i,) for i in range(data.data.nbL)]
        lprefix = lprefix + [(i, j) for i in range(data.data.nbL)
                             for j in range(data.data.nbL)]
        lsuffix = [()]
        lsuffix = lsuffix + [(i,) for i in range(data.data.nbL)]
        lsuffix = lsuffix + [(i, j) for i in range(data.data.nbL)
                             for j in range(data.data.nbL)]

        for s in data.data.pref.keys():
            self.assertTrue(s in lprefix)
        for s in data.data.suff.keys():
            self.assertTrue(s in lsuffix)
        l_h = Hankel(sample_instance=data.data, lrows=1, lcolumns=1,
                     version="prefix", partial=False,
                     sparse=False).lhankel
        self.assertEqual(3, l_h.__len__())
        h_0 = np.array([[ 6.,  3.,  2.,  2.,  1.],
                        [ 3.,  0.,  2.,  0.,  0.],
                        [ 2.,  0.,  1.,  0.,  0.],
                        [ 2.,  0.,  0.,  0.,  0.],
                        [ 1.,  0.,  0.,  0.,  0.]])


        h_1 = np.array([[ 3.,  0.,  2.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.]])


        h_2 = np.array([[ 2.,  0.,  1.,  0.,  0.],
                        [ 2.,  0.,  0.,  0.,  0.],
                        [ 1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.]])


        np.testing.assert_equal(l_h[0], h_0)
        np.testing.assert_equal(l_h[1], h_1)
        np.testing.assert_equal(l_h[2], h_2)

        l_h = Hankel(sample_instance=data.data, lrows=1, lcolumns=1,
                     version="prefix", partial=False, sparse=True).lhankel
        self.assertEqual(3, l_h.__len__())
        h_0 = sps.dok_matrix(np.array([[ 6.,  3.,  2.,  2.,  1.],
                        [ 3.,  0.,  2.,  0.,  0.],
                        [ 2.,  0.,  1.,  0.,  0.],
                        [ 2.,  0.,  0.,  0.,  0.],
                        [ 1.,  0.,  0.,  0.,  0.]]))

        h_1 = sps.dok_matrix(np.array([[ 3.,  0.,  2.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.]]))

        h_2 = sps.dok_matrix(np.array([[ 2.,  0.,  1.,  0.,  0.],
                        [ 2.,  0.,  0.,  0.,  0.],
                        [ 1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.]]))

        np.testing.assert_equal(l_h[0], h_0)
        np.testing.assert_equal(l_h[1], h_1)
        np.testing.assert_equal(l_h[2], h_2)

    def test_CreateHankel_suffix(self):
        adr = get_dataset_path("essai")
        # adr = get_dataset_path("3.pautomac.train")
        data = load_data_sample(adr=adr)
        cl = Spectral(partial=False, version="suffix")
        cl.polulate_dictionnaries(data.data)
        lprefix = [()]
        lprefix = lprefix + [(i,) for i in range(data.data.nbL)]
        lprefix = lprefix + [(i, j) for i in range(data.data.nbL)
                             for j in range(data.data.nbL)]
        lsuffix = [()]
        lsuffix = lsuffix + [(i,) for i in range(data.data.nbL)]
        lsuffix = lsuffix + [(i, j) for i in range(data.data.nbL)
                             for j in range(data.data.nbL)]

        for s in data.data.pref.keys():
            self.assertTrue(s in lprefix)
        for s in data.data.suff.keys():
            self.assertTrue(s in lsuffix)
        l_h = Hankel(sample_instance=data.data, lrows=1, lcolumns=1,
                     version="suffix", partial=False,
                     sparse=False).lhankel
        self.assertEqual(3, l_h.__len__())
        h_0 = np.array([[6., 1., 4., 2., 1.],
                        [1., 0., 2., 0., 0.],
                        [4., 0., 1., 0., 0.],
                        [2., 0., 0., 0., 0.],
                        [1., 0., 0., 0., 0.]])

        h_1 = np.array([[1., 0., 2., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.]])

        h_2 = np.array([[4., 0., 1., 0., 0.],
                        [2., 0., 0., 0., 0.],
                        [1., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.]])

        np.testing.assert_equal(l_h[0], h_0)
        np.testing.assert_equal(l_h[1], h_1)
        np.testing.assert_equal(l_h[2], h_2)

        l_h = Hankel(sample_instance=data.data, lrows=1, lcolumns=1,
                     version="suffix", partial=False, sparse=True).lhankel
        self.assertEqual(3, l_h.__len__())
        h_0 = sps.dok_matrix(np.array([[6., 1., 4., 2., 1.],
                                       [1., 0., 2., 0., 0.],
                                       [4., 0., 1., 0., 0.],
                                       [2., 0., 0., 0., 0.],
                                       [1., 0., 0., 0., 0.]]))

        h_1 = sps.dok_matrix(np.array([[1., 0., 2., 0., 0.],
                                       [0., 0., 0., 0., 0.],
                                       [0., 0., 0., 0., 0.],
                                       [0., 0., 0., 0., 0.],
                                       [0., 0., 0., 0., 0.]]))

        h_2 = sps.dok_matrix(np.array([[4., 0., 1., 0., 0.],
                                       [2., 0., 0., 0., 0.],
                                       [1., 0., 0., 0., 0.],
                                       [0., 0., 0., 0., 0.],
                                       [0., 0., 0., 0., 0.]]))

        np.testing.assert_equal(l_h[0], h_0)
        np.testing.assert_equal(l_h[1], h_1)
        np.testing.assert_equal(l_h[2], h_2)

    def test_CreateHankel_factor(self):
        adr = get_dataset_path("essai")
        # adr = get_dataset_path("3.pautomac.train")
        data = load_data_sample(adr=adr)
        cl = Spectral(partial=False, version="factor")
        cl.polulate_dictionnaries(data.data)
        lprefix = [()]
        lprefix = lprefix + [(i,) for i in range(data.data.nbL)]
        lprefix = lprefix + [(i, j) for i in range(data.data.nbL)
                             for j in range(data.data.nbL)]
        lsuffix = [()]
        lsuffix = lsuffix + [(i,) for i in range(data.data.nbL)]
        lsuffix = lsuffix + [(i, j) for i in range(data.data.nbL)
                             for j in range(data.data.nbL)]

        for s in data.data.pref.keys():
            self.assertTrue(s in lprefix)
        for s in data.data.suff.keys():
            self.assertTrue(s in lsuffix)
        l_h = Hankel(sample_instance=data.data, lrows=1, lcolumns=1,
                     version="factor", partial=False,
                     sparse=False).lhankel
        self.assertEqual(3, l_h.__len__())
        h_0 = np.array([[14., 3., 5., 2., 1.],
                        [3., 0., 2., 0., 0.],
                        [5., 0., 1., 0., 0.],
                        [2., 0., 0., 0., 0.],
                        [1., 0., 0., 0., 0.]])

        h_1 = np.array([[3., 0., 2., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.]])

        h_2 = np.array([[5., 0., 1., 0., 0.],
                        [2., 0., 0., 0., 0.],
                        [1., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.]])

        np.testing.assert_equal(l_h[0], h_0)
        np.testing.assert_equal(l_h[1], h_1)
        np.testing.assert_equal(l_h[2], h_2)

        l_h = Hankel(sample_instance=data.data, lrows=1, lcolumns=1,
                     version="factor", partial=False, sparse=True).lhankel
        self.assertEqual(3, l_h.__len__())
        h_0 = sps.dok_matrix(np.array([[14., 3., 5., 2., 1.],
                                       [3., 0., 2., 0., 0.],
                                       [5., 0., 1., 0., 0.],
                                       [2., 0., 0., 0., 0.],
                                       [1., 0., 0., 0., 0.]]))

        h_1 = sps.dok_matrix(np.array([[3., 0., 2., 0., 0.],
                                       [0., 0., 0., 0., 0.],
                                       [0., 0., 0., 0., 0.],
                                       [0., 0., 0., 0., 0.],
                                       [0., 0., 0., 0., 0.]]))

        h_2 = sps.dok_matrix(np.array([[5., 0., 1., 0., 0.],
                                       [2., 0., 0., 0., 0.],
                                       [1., 0., 0., 0., 0.],
                                       [0., 0., 0., 0., 0.],
                                       [0., 0., 0., 0., 0.]]))

        np.testing.assert_equal(l_h[0], h_0)
        np.testing.assert_equal(l_h[1], h_1)
        np.testing.assert_equal(l_h[2], h_2)

    def test_Attr_Hankel(self):
        adr = get_dataset_path("essai")
        # adr = get_dataset_path("3.pautomac.train")
        data = load_data_sample(adr=adr)
        cl = Spectral()
        cl.polulate_dictionnaries(data.data)
        h = Hankel(sample_instance=data.data, lrows=1, lcolumns=1,
            version="classic", partial=False, sparse=False)
        with self.assertRaises(TypeError):
            h.nbL = 2.0
        # nbEx assignment
        with self.assertRaises(TypeError):
            h.nbEx = 6.8

    def testEqualityOperator(self):
        adr = get_dataset_path("essai")
        data = load_data_sample(adr=adr)
        cl = Spectral()
        cl.polulate_dictionnaries(data.data)
        h1 = Hankel(sample_instance=data.data, lrows=1, lcolumns=1,
                    version="classic", partial=False, sparse=False)
        h2 = Hankel(sample_instance=data.data, lrows=1, lcolumns=1,
                     version="prefix", partial=False, sparse=False)
        self.assertTrue(h1 != h2)
        h2 = Hankel(sample_instance=data.data, lrows=1, lcolumns=1,
                    version="classic", partial=True, sparse=False)
        self.assertTrue(h1 != h2)
        h2 = Hankel(sample_instance=data.data, lrows=1, lcolumns=1,
                    version="classic", partial=False, sparse=True)
        self.assertTrue(h1 != h2)
        h2 = Hankel(sample_instance=data.data, lrows=1, lcolumns=1,
                    version="classic", partial=False, sparse=False)
        h2.lhankel[0][0,0] -= 1.0
        self.assertTrue(h1 != h2)
        h2.lhankel = h2.lhankel[1:]
        self.assertTrue(h1 != h2)
        h2.nbEx -= 1
        self.assertTrue(h1 != h2)
        h2.nbL -= 1
        self.assertTrue(h1 != h2)
        l = [(0,), (1,), (0,0), (0,1), (1,0), (1,1)]
        h2 = h1.to_automaton(2).to_hankel(lrows=l, lcolumns=l)
        self.assertTrue(h1 != h2)
        
        adr = get_dataset_path("3.pautomac.train")
        data = load_data_sample(adr=adr)
        X = data.data
        sp1 = Spectral()
        sp1 = sp1.fit(X)
        H1 = sp1.hankel
        sp2 = Spectral()
        sp2 = sp2.fit(X)
        H2 = sp2.hankel
        H2.lhankel[0][(1, 1610)] = 0
        self.assertTrue(H1 != H2)
        

if __name__ == '__main__':
    unittest.main()
