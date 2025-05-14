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
import unittest
import numpy as np
import os
from collections import deque
import yaml

from splearn.automaton import Automaton
from splearn.hankel import Hankel
from splearn.serializer import Serializer
from splearn.spectral import Spectral
from splearn.tests.datasets.get_dataset_path import get_dataset_path
from splearn.datasets.base import load_data_sample

class SerializerTest(unittest.TestCase):


    def setUp(self):
        self.A = Automaton.SimpleExample()
        self.title = "Simple Example"
        self.input_file = 'simple_example'
        self.formats = ['json', 'yaml']
        self.words = [[], [0], [0, 0], [1], [1, 1], [0, 1, 0], [1, 0, 1]]
        self.d_str = "Value for word \"{:s}\" = {:.2f}"
  
    def testReadAutomaton(self):
        for f in self.formats:
            B = Automaton.read(get_dataset_path(self.input_file + '.' + f), format=f)
            for w in self.words:
                np.testing.assert_almost_equal(self.A.val(w), B.val(w))
    
    def testWriteAutomata(self):
        for f in self.formats:
            Automaton.write(self.A, get_dataset_path(self.input_file + '_2.' + f), format=f)
            B = Automaton.read(get_dataset_path(self.input_file + '_2.' + f), format=f)
            for w in self.words:
                np.testing.assert_almost_equal(self.A.val(w), B.val(w))
        for f in self.formats:
            os.remove(get_dataset_path(self.input_file + '_2.' + f))
    
    def testReadHankel(self):
        for f in self.formats:
            H = self.A.to_hankel([(), (0,), (1,)], [(), (0,), (1,)])
            Hankel.write(H, get_dataset_path(self.input_file + "_hankel" + "." + f), format=f)
            Hb = Hankel.read(get_dataset_path(self.input_file + "_hankel" + "." + f), format = f)
            B = Hb.to_automaton(2)
            for w in self.words:
                np.testing.assert_almost_equal(self.A.val(w), B.val(w))
        for f in self.formats:
            os.remove(get_dataset_path(self.input_file + "_hankel" + "." + f))
    
    def testReadWriteRealHankel(self):
        adr = get_dataset_path("3.pautomac.train")
        data = load_data_sample(adr=adr)
        X = data.data
        sp = Spectral()
        sp = sp.fit(X)
        H = sp.hankel
        for f in self.formats:
            Hankel.write(H, get_dataset_path("3.pautomac.train" + "_hankel" + "." + f), format=f)
            Hb = Hankel.read(get_dataset_path("3.pautomac.train" + "_hankel" + "." + f), format = f)
            self.assertTrue(H == Hb)
        for f in self.formats:
            os.remove(get_dataset_path("3.pautomac.train" + "_hankel" + "." + f))
    
    def testOthersSerializationTypes(self):
        data = [{'a' : 10, 40 : 'gu'}, {'toto', 5, 2.5, 'b'}, ('gh', 25, 'ko', 1.0)]
        data_json_str = Serializer.data_to_json(data)
        data_yaml_str = Serializer.data_to_yaml(data)
        data_json = Serializer.json_to_data(data_json_str)
        data_yaml = Serializer.yaml_to_data(data_yaml_str)
        self.assertEqual(data, data_json)
        self.assertEqual(data, data_yaml)
        
        data = [1, 2, 3.0]
        data_json_str = Serializer.data_to_json(data)
        data_yaml_str = Serializer.data_to_yaml(data)
        data_json = Serializer.json_to_data(data_json_str)
        data_yaml = Serializer.yaml_to_data(data_yaml_str)
        self.assertEqual(data, data_json)
        self.assertEqual(data, data_yaml)
    
    def testBadTypeSerialieationException(self):
        with self.assertRaises(TypeError):
            Serializer.data_to_json(deque('ghi'))
    
    def testBadDataException(self):
        yamlstrList = ["- scipy.dok_matrix:", 
                       "- scipy.dok_matrix:\n    shape:\n        tuple: [1, 1]\n    values: {'(0,0)': 1.0}",
                       "- numpy.ndarray:",
                       "- numpy.ndarray:\n    dtype: float64",
                       "- automaton:",
                       "- automaton:\n    nbL: 1",
                       "- hankel:",
                       "- hankel:\n    nbL: 1"]
        jsonstrList = ["{\"scipy.dok_matrix\":{}}",
                       "{\"numpy.ndarray\":{}}",
                       "{\"automaton\":{}}",
                       "{\"hankel\":{}}"]
        for yamlstr in yamlstrList:
            with self.assertRaises(ValueError):
                Serializer.yaml_to_data(yamlstr)
        for jsonstr in jsonstrList:
            with self.assertRaises(ValueError):
                Serializer.json_to_data(jsonstr)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()