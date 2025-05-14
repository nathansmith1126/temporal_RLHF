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
import math

from splearn.automaton import Automaton
from splearn.tests.datasets.get_dataset_path import get_dataset_path
from splearn.datasets.base import load_data_sample
from splearn import Spectral

class UnitaryTest(unittest.TestCase):

    def setUp(self):
        self.A = Automaton(2, 2, np.array([1.0, -1.0]),
                           np.array([0.0, 1.0]),
                           [np.array([[0.5, 1.0], [-0.5, 0.0]]),
                           np.array([[1.0, 1.0], [-2.0, 1/3]])])

    def test_constructor(self):
        # final assignment
        self.assertRaises(TypeError, Automaton, 2, 2, np.array([1.0, -1.0]),
                          "final", [np.array([[0.5, 1.0], [-0.5, 0.0]]),
                                    np.array([[1.0, 1.0], [-2.0, 1/3]])])
        self.assertRaises(TypeError, Automaton, 2, 2, np.array([1.0, -1.0]),
                          np.array([0, 1]), [np.array([[0.5, 1.0],
                                                       [-0.5, 0.0]]),
                                             np.array([[1.0, 1.0],
                                                       [-2.0, 1/3]])])
        self.assertRaises(ValueError, Automaton, 2, 2, np.array([1.0, -1.0]),
                          np.array([0.0, 1.0, 1.0]), [np.array([[0.5, 1.0],
                                                                [-0.5, 0.0]]),
                                                      np.array([[1.0, 1.0],
                                                                [-2.0, 1/3]])])
        # initial assignment
        self.assertRaises(TypeError, Automaton, 2, 2, "np.array([1.0, -1.0])",
                          np.array([0.0, 1.0]), [np.array([[0.5, 1.0],
                                                           [-0.5, 0.0]]),
                                                 np.array([[1.0, 1.0],
                                                           [-2.0, 1/3]])])
        self.assertRaises(TypeError, Automaton, 2, 2, np.array([1, -1]),
                          np.array([0.0, 1.0]), [np.array([[0.5, 1.0],
                                                           [-0.5, 0.0]]),
                                                 np.array([[1.0, 1.0],
                                                           [-2.0, 1/3]])])
        self.assertRaises(ValueError, Automaton, 2, 2,
                          np.array([1.0, -1.0, 0.0]),
                          np.array([0.0, 1.0]), [np.array([[0.5, 1.0],
                                                           [-0.5, 0.0]]),
                                                 np.array([[1.0, 1.0],
                                                           [-2.0, 1/3]])])
        # transitions assignment
        self.assertRaises(TypeError, Automaton, 2, 2, np.array([1.0, -1.0]),
                          np.array([0.0, 1.0]), "transitions")
        self.assertRaises(ValueError, Automaton, 2, 2, np.array([1.0, -1.0]),
                          np.array([0.0, 1.0]), [1, 2, 3])
        self.assertRaises(TypeError, Automaton, 2, 2, np.array([1.0, -1.0]),
                          np.array([0.0, 1.0]), [1, 2])
        self.assertRaises(TypeError, Automaton, 2, 2, np.array([1.0, -1.0]),
                          np.array([0.0, 1.0]), [np.array([[0, 1],
                                                           [0, 0]]),
                                                 np.array([[1, 1],
                                                           [2, 1]])])
        self.assertRaises(ValueError, Automaton, 2, 2, np.array([1.0, -1.0]),
                          np.array([0.0, 1.0]), [np.ones((2, 2, 2),
                                                         dtype=np.float),
                                                 np.ones((2, 2, 2),
                                                         dtype=np.float)])
        self.assertRaises(ValueError, Automaton, 2, 2, np.array([1.0, -1.0]),
                          np.array([0.0, 1.0]), [np.ones((2, 3),
                                                         dtype=np.float),
                                                 np.ones((2, 3),
                                                         dtype=np.float)])
        self.assertRaises(ValueError, Automaton, 2, 2, np.array([1.0, -1.0]),
                          np.array([0.0, 1.0]), [np.ones((3, 3),
                                                         dtype=np.float),
                                                 np.ones((3, 3),
                                                         dtype=np.float)])
        # nbS assignment
        self.assertRaises(ValueError, Automaton, 2, 2.4, np.array([1.0, -1.0]),
                          np.array([0.0, 1.0]), [np.array([[0.5, 1.0],
                                                           [-0.5, 0.0]]),
                                                 np.array([[1.0, 1.0],
                                                           [-2.0, 1/3]])])
        self.assertRaises(ValueError, Automaton, 2, -2, np.array([1.0, -1.0]),
                          np.array([0.0, 1.0]), [np.array([[0.5, 1.0],
                                                           [-0.5, 0.0]]),
                                                 np.array([[1.0, 1.0],
                                                           [-2.0, 1/3]])])
        # nbL assignment
        self.assertRaises(ValueError, Automaton, 2.4, 2, np.array([1.0, -1.0]),
                          np.array([0.0, 1.0]), [np.array([[0.5, 1.0],
                                                           [-0.5, 0.0]]),
                                                 np.array([[1.0, 1.0],
                                                           [-2.0, 1/3]])])
        self.assertRaises(ValueError, Automaton, -2, 2, np.array([1.0, -1.0]),
                          np.array([0.0, 1.0]), [np.array([[0.5, 1.0],
                                                           [-0.5, 0.0]]),
                                                 np.array([[1.0, 1.0],
                                                           [-2.0, 1/3]])])

    def test_initial(self):
        np.testing.assert_array_equal(self.A.initial, np.array([1.0, -1.0]))

    def test_final(self):
        np.testing.assert_array_equal(self.A.final, np.array([0.0, 1.0]))

    def test_transitions(self):
        np.testing.assert_array_equal(self.A.transitions[0],
                                      np.array([[0.5, 1.0], [-0.5, 0.0]]))
        np.testing.assert_array_equal(self.A.transitions[1],
                                      np.array([[1.0, 1.0], [-2.0, 1/3]]))

    def test_nbS(self):
        self.assertEqual(self.A.nbS, 2)

    def test_nbL(self):
        self.assertEqual(self.A.nbL, 2)

    def test__rmul(self):
        B = 3.1 * self.A
        np.testing.assert_array_equal(B.initial, np.array([3.1, -3.1]))

    def test__add(self):
        # Test errors
        Ap = Automaton(3, 2, np.array([1.0, -1.0]),
                       np.array([0.0, 1.0]),
                       [np.array([[0.5, 1.0], [-0.5, 0.0]]),
                        np.array([[1.0, 1.0], [-2.0, 1/3]]),
                        np.array([[-1.0, 1.0], [-1.0, 1/6]])])
        self.assertRaises(ValueError, self.A.__add__, Ap)
        # Test addition validity
        B = self.A + self.A
        self.assertEqual(B.nbL, 2)
        self.assertEqual(B.nbS, 4)
        np.testing.assert_array_equal(B.initial,
                                      np.array([1.0, -1.0, 1.0, -1.0]))
        np.testing.assert_array_equal(B.final, np.array([0.0, 1.0, 0.0, 1.0]))
        np.testing.assert_array_equal(B.transitions[0],
                                      np.array([[0.5, 1.0, 0.0, 0.0],
                                                [-0.5, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.5, 1.0],
                                                [0.0, 0.0, -0.5, 0.0]]))

    def test_val(self):
        np.testing.assert_almost_equal(self.A.val([]), -1.0)
        np.testing.assert_almost_equal(self.A.val([0]), 1.0)
        np.testing.assert_almost_equal(self.A.val([1]), 2/3)
        np.testing.assert_almost_equal(self.A.val([0, 1]), 4/3)

    def test_mirror(self):
        B = self.A
        C = B.mirror()
        u = np.random.randint(0, B.nbL, 5)
        np.testing.assert_almost_equal(B.val(u), C.val(u[::-1]))

    def test_HouseholderReflector(self):
        v = np.array([3, 4])
        v = Automaton.HouseholderReflector(v)
        r = math.sqrt(5)/5*np.array([2, 1])
        np.testing.assert_array_almost_equal_nulp(v, r)
        v = np.array([-3, 4])
        v = Automaton.HouseholderReflector(v)
        r = math.sqrt(5) / 5*np.array([-2, 1])
        np.testing.assert_array_almost_equal_nulp(v, r)

    def test_mulHouseholderReflector(self):
        u = np.array([1, 2, 3])
        v = np.array([1, 2])
        w = Automaton.mulHouseholderReflector(u, v)
        np.testing.assert_array_almost_equal_nulp(w, np.array([1, -14, -29]))

    def test_HouseholderReductionFw(self):
        nbe = 2
        nbl = 2
        i = np.random.rand(nbe)
        t = np.random.rand(nbe)
        trans = [np.random.rand(nbe, nbe) for _ in range(nbl)]
        newNbe = 10
        p = np.random.rand(nbe, newNbe)
        ip = np.linalg.pinv(p)

        A = Automaton(nbl, newNbe, np.dot(i, p), np.dot(ip, t),
                      [np.dot(ip, np.dot(t, p)) for t in trans])
        B = A.HouseholderReductionFw(1.0e-5)
        # pr int(B.nbL,B.nbS,B.initial,B.final,B.transitions)
        self.assertEqual(B.nbS, nbe)
        np.testing.assert_almost_equal(A.val([]), B.val([]))
        np.testing.assert_almost_equal(A.val([0]), B.val([0]))
        np.testing.assert_almost_equal(A.val([0, 0]), B.val([0, 0]))
        np.testing.assert_almost_equal(A.val([1]), B.val([1]))
        np.testing.assert_almost_equal(A.val([0, 1, 0]), B.val([0, 1, 0]))
        np.testing.assert_almost_equal(A.val([0, 1, 0, 1, 1]),
                                       B.val([0, 1, 0, 1, 1]))

    def test_minimisation(self):
        nbe = 2
        nbl = 2
        i = np.random.rand(nbe)
        t = np.random.rand(nbe)
        trans = [np.random.rand(nbe, nbe) for _ in range(nbl)]
        newNbe = 10
        p = np.random.rand(nbe, newNbe)
        ip = np.linalg.pinv(p)

        A = Automaton(nbl, newNbe, np.dot(i, p), np.dot(ip, t),
                      [np.dot(ip, np.dot(t, p)) for t in trans])
        B = Automaton(nbl, nbe, np.zeros(nbe), np.random.rand(nbe),
                      [np.random.rand(nbe, nbe) for _ in range(nbl)])
        B = A+B
        C = Automaton(nbl, nbe, np.random.rand(nbe), np.zeros(nbe),
                      [np.random.rand(nbe, nbe) for _ in range(nbl)])
        B = B+C
        B = B.minimisation(1.0e-5)

        self.assertEqual(B.nbS, nbe)
        np.testing.assert_almost_equal(A.val([]), B.val([]))
        np.testing.assert_almost_equal(A.val([0]), B.val([0]))
        np.testing.assert_almost_equal(A.val([0, 0]), B.val([0, 0]))
        np.testing.assert_almost_equal(A.val([1]), B.val([1]))
        np.testing.assert_almost_equal(A.val([0, 1, 0]), B.val([0, 1, 0]))
        np.testing.assert_almost_equal(A.val([0, 1, 0, 1, 1]),
                                       B.val([0, 1, 0, 1, 1]))

    def test_absConv(self):
        nbe = 3
        nbl = 2
        i = np.random.rand(nbe)
        t = np.random.rand(nbe)
        trans = [np.random.rand(nbe, nbe)]
        trans.append(np.eye(nbe))
        A = Automaton(nbl, nbe, i, t, trans)
        self.assertEqual(A.isAbsConv, False)
        trans = [np.random.rand(nbe, nbe) for x in range(nbl)]
        i = i/i.sum()
        for q in range(nbe):
            s = t[q]
            for x in range(nbl):
                s += sum(trans[x][q, :])
            t[q] = t[q]/s
            for x in range(nbl):
                trans[x][q, :] = trans[x][q, :]/s
        A = Automaton(nbl, nbe, i, t, trans)
        self.assertEqual(A.isAbsConv, True)

    def test_sum(self):
        nbe = 3
        nbl = 2
        i = np.random.rand(nbe)
        t = np.random.rand(nbe)
        trans = [np.random.rand(nbe, nbe) for x in range(nbl)]
        i = i/i.sum()
        for q in range(nbe):
            s = t[q]
            for x in range(nbl):
                s += sum(trans[x][q, :])
            t[q] = t[q]/s
            for x in range(nbl):
                trans[x][q, :] = trans[x][q, :]/s
        A = Automaton(nbl, nbe, i, t, trans)
        np.testing.assert_almost_equal(A.sum(), 1)

    def test_load_Pautomac_Automaton(self):
        adr = get_dataset_path("pautomac3.txt")
        A = Automaton.load_Pautomac_Automaton(adr)
        r = A.nbL, A.nbS
        self.assertEqual(r, (4, 25))  # à compléter

    def test_SimpleExample(self):
        A = Automaton.SimpleExample()
        np.testing.assert_almost_equal(A.sum(), 1)

    def test_BuildHankels(self):
        A = Automaton.SimpleExample()
        lh = A.BuildHankels([(), (0,), (1,)],
                            [(), (0,), (1,)])
        self.assertEqual(lh[0][0, 0], 1/3)

    def test_transformation(self):
        A1 = Automaton.SimpleExample()
        t = A1.transformation(source='classic',target='prefix')
        A2 = t.transformation(source='prefix',target='classic')
        np.testing.assert_almost_equal(A1.final, A2.final)
        np.testing.assert_almost_equal(A1.initial, A2.initial)
        np.testing.assert_almost_equal(A1.transitions, A2.transitions)

    def test_calc_prefix_completion_weights(self):
        A = Automaton.SimpleExample()
        A = A.transformation()
        dic = A.calc_prefix_completion_weights([-1])
        self.assertEqual(set(dic.keys()), {0, 1, -1},
                         "test_calc_prefix_completion_weights failed")
    
    def test_get_dot(self):        
        train_file = '3.pautomac_light.train'
        data = load_data_sample(adr=get_dataset_path(train_file))
        sp = Spectral()
        sp.fit(X=data.data)
        dotfile = "3.pautomac_light.train.dot"
        dot = sp.automaton.get_dot(threshold = 0.2, title = dotfile)
        print(dot)
        gold_file = train_file + ".gv"
        gold_str = ""
        with open(get_dataset_path(gold_file), 'r') as f:
            gold_str = f.read()
        self.assertEqual(dot, gold_str, "test_get_dot failed")

if __name__ == '__main__':
    unittest.main()
