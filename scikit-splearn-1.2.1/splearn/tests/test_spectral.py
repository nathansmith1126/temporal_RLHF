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
from splearn.automaton import Automaton
from splearn.spectral import Spectral
from splearn.tests.datasets.get_dataset_path import get_dataset_path
from sklearn.linear_model.tests.test_passive_aggressive import random_state


class SpectralTest(unittest.TestCase):

    def test_version(self):
        adr = get_dataset_path("essai")
        data = load_data_sample(adr=adr)
        cl = Spectral(partial=False, sparse=False, full_svd_calculation= True, version="prefix")
        X = data.data
        cl.fit(X=X)
        np.testing.assert_almost_equal(cl.automaton.initial,
                               np.array([-1.21159819e+00, -1.78488894e-01,
                                         1.30974719e-02, -1.66533454e-16,
                                         -3.70074342e-17]))

        cl = Spectral(partial=False, sparse=False, full_svd_calculation= True, version="suffix")
        cl.fit(X=X)
        np.testing.assert_almost_equal(cl.automaton.initial,
                                       np.array(
                                           [-3.16481225e-01, -1.95397796e-01,
                                            2.89623130e-01, 1.85499764e-15,
                                            0.00000000e+00]))
        cl = Spectral(partial=False, sparse=False,
                      version="factor", full_svd_calculation= True, mode_quiet=False)
        cl.fit(X=X)
        np.testing.assert_almost_equal(cl.automaton.final,
                                       np.array([ -4.35303631e-01,
                                                  1.49070953e+00,
                                                  4.80783716e-01,
                                                  3.75818126e-17,
                                                  0.00000000e+00]))

    def test_rank_fail(self):
        adr = get_dataset_path("essai")
        data = load_data_sample(adr=adr)
        X = data.data
        cl = Spectral(partial=True, lrows= 5, lcolumns=5,
                      sparse=False, version="prefix", rank=7)
        with self.assertRaises(ValueError):
            cl.fit(X=X)


    def test_non_automaton(self):
        cl = Spectral(partial=False, sparse=False)
        param = cl.get_params()
        self.assertEqual(param.get("partial"),False)
        X = np.array([[1, 2, 3, 6],[0, 0, 2, 5]])
        cl.fit(X=X)
        self.assertEqual(cl.automaton, None)

    def test_fit_spectral_notsparse(self):
        adr = get_dataset_path("essai")
        # adr = get_dataset_path("3.pautomac.train")
        data = load_data_sample(adr=adr)
        cl = Spectral(partial=False, sparse=False)
        cl.fit(X=data.data)
        self.assertIsInstance(cl.automaton,Automaton,"fit function fails")

    def test_fit_spectral_sparse(self):
        adr = get_dataset_path("essai")
        # adr = get_dataset_path("3.pautomac.train")
        data = load_data_sample(adr=adr)
        cl = Spectral(partial=False, sparse=True, rank=4)
        cl.fit(X=data.data)
        self.assertIsInstance(cl.automaton,Automaton,"fit function fails")


    def test_predict_spectral(self):
        adr = get_dataset_path("essai")
        # adr = get_dataset_path("3.pautomac.train")
        data = load_data_sample(adr=adr)
        cl = Spectral(partial=False, sparse=False)
        cl.fit(X=data.data)
        XX = data.data[0:3,:]
        p = cl.predict(X=XX)
        np.testing.assert_almost_equal(
            p[0], np.array([ 0.16666667,  0.16666667,  0.16666667]))

    def test_loss_spectral(self):
        adr = get_dataset_path("essai")
        data = load_data_sample(adr=adr)
        cl = Spectral(partial=False, sparse=False, smooth_method = 'trigram')
        cl.fit(X=data.data)
        XX = data.data[0:3, :]
        p = cl.loss(X=XX, normalize=False)
        self.assertAlmostEquals(p, 5.375278407684168)
        p = cl.loss(X=XX, normalize=True)
        self.assertAlmostEquals(p, 1.7917594692280561)

    def test_score_spectral(self):
        adr = get_dataset_path("essai")
        data = load_data_sample(adr=adr)
        cl = Spectral(partial=False, sparse=False, smooth_method = 'trigram')
        cl.fit(X=data.data)
        XX = data.data[0:3, :]
        p = cl.score(X=XX, scoring="perplexity")
        self.assertAlmostEquals(p, -1.7917594692280561)
        p = cl.score(X=XX, scoring="none")
        self.assertAlmostEquals(p, -1.7917594692280561)


    def test_trigram(self):
        adr = get_dataset_path("3.pautomac.train")
        data = load_data_sample(adr=adr)
        # adr = get_dataset_path("essai")
        cl = Spectral(lrows=6, lcolumns=6, sparse=False,
                      partial=True)
        cl.set_params(smooth_method='trigram')
        cl.fit(data.data)
        self.assertEqual(cl.trigram[(1, 2)],
                         {0: 1543, 1: 1278, 2: 453, 3: 1254, -1: 5060, -2: 532} )
        with self.assertRaises(TypeError):
            cl.trigram = "dict"
        adr = get_dataset_path("essai")
        data = load_data_sample(adr=adr)
        cl = Spectral(lrows=6, lcolumns=6, sparse=False,
                      partial=True, smooth_method='trigram')
        cl.fit(data.data)
        cl.predict(data.data)
        np.testing.assert_equal(cl.trigram_index,
                                np.zeros(data.data.shape[0], dtype=bool))

        self.assertEqual(cl.nb_trigram(), 0)

    def test_trigramprobability(self):
        adr = get_dataset_path("3.pautomac.train")
        data = load_data_sample(adr=adr)
        # adr = get_dataset_path("essai")
        cl = Spectral(lrows=6, lcolumns=6, sparse=False,
                      partial=True)
        cl.set_params(smooth_method='trigram')
        cl.fit(data.data)
        lo = cl.loss(X=data.data)
        self.assertAlmostEqual(lo, 10.875367424508504)
        cl.nb_trigram()

    def test_loss_with_y(self):
        adr = get_dataset_path("3.pautomac.train")
        data = load_data_sample(adr=adr)
        # adr = get_dataset_path("essai")
        cl = Spectral(lrows=6, lcolumns=6, sparse=False,
                      partial=True)
        cl.set_params(smooth_method='trigram')
        cl.fit(data.data)
        p = cl.predict(X=data.data)
        lo = cl.loss(X=data.data, y=p, normalize=True)
        self.assertAlmostEqual(lo, 0.0)
        lo = cl.loss(X=data.data, y=p, normalize=False)
        self.assertAlmostEqual(lo, 0.0)


    def test_score_with_y(self):
        adr = get_dataset_path("3.pautomac.train")
        data = load_data_sample(adr=adr)
        # adr = get_dataset_path("essai")
        cl = Spectral(lrows=6, lcolumns=6, sparse=False,
                      partial=True)
        cl.set_params(smooth_method='trigram')
        cl.fit(data.data)
        p = cl.predict(X=data.data)
        lo = cl.score(X=data.data, y=p, scoring='perplexity')
        self.assertAlmostEqual(lo, 6220.871643894891)
        lo = cl.score(X=data.data, y=p, scoring='none')
        self.assertAlmostEqual(lo, -0.0)

    def test_smooth_method_none(self):
        adr = get_dataset_path("3.pautomac.train")
        data = load_data_sample(adr=adr)
        # adr = get_dataset_path("essai")
        cl = Spectral(lrows=6, lcolumns=6, sparse=False,
                      partial=True)
        cl.fit(data.data)
        p = cl.predict(X=data.data)
        with self.assertRaises(ValueError):
            cl.score(X=data.data)
        with self.assertRaises(ValueError):
            cl.score(X=data.data, y=p)
        self.assertAlmostEqual(cl.loss(X=data.data, y=p), 0.0)

    def test_smooth_method_notnone(self):
        with self.assertWarns(UserWarning):
            cl = Spectral(lrows=6, lcolumns=6, sparse=False,
                      partial=True, smooth_method='weighted')


    def test_BuildAutomatonFromHankel(self):  # a suivre
        A = Automaton.SimpleExample()
        H = A.to_hankel([(), (0,), (1,)], [(), (0,), (1,)], mode_quiet=False)
        B = H.to_automaton(rank=2, mode_quiet=False)
        np.testing.assert_almost_equal(A.val([]), B.val([]))
        np.testing.assert_almost_equal(A.val([0]), B.val([0]))
        np.testing.assert_almost_equal(A.val([0, 0]), B.val([0, 0]))
        np.testing.assert_almost_equal(A.val([1]), B.val([1]))
        np.testing.assert_almost_equal(A.val([0, 1, 0]), B.val([0, 1, 0]))
        np.testing.assert_almost_equal(A.val([0, 1, 0, 1, 1]),
                                       B.val([0, 1, 0, 1, 1]))

    def test_sklearn_compatibility(self):
        from sklearn.utils.estimator_checks import check_estimator
        from sklearn.model_selection import train_test_split, cross_val_score
        check_estimator(Spectral)
        adr = get_dataset_path("3.pautomac_light.train")
        data = load_data_sample(adr=adr)
        sp = Spectral(lrows=6, lcolumns=6, rank = 5, sparse=False,
                      partial=True, smooth_method='trigram')
        X_train, X_test = train_test_split(data.data, test_size=0.4, random_state=0)
        sp.fit(X_train)
        single_predicted_weights = sp.predict(X_test)
        print(single_predicted_weights)
        self.assertAlmostEqual(single_predicted_weights[0], 6.76217667e-02, delta = 1e-5)
        scores = cross_val_score(sp, data.data, cv=4)
        print(scores)
        scores_expected = [-10.65272755, -10.7090267,  -10.78404758, -11.08453211]
        for s1, s2 in zip(scores, scores_expected):
            self.assertAlmostEqual(s1, s2, delta=0.1)

#     def test_Perplexity(self):
#         adr = get_dataset_path("3.pautomac")
#         P = Learning()
#         P.load_Spice_Sample(adr + ".train")
#         A = P.LearnAutomaton(20, 7, 7, "factor", True)
#         P.Perplexity(A, adr)

if __name__ == '__main__':
    unittest.main()
