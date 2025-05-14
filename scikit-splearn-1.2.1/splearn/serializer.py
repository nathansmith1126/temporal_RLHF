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
"""This module contains the Serializer class
"""
import numpy as np

from splearn.automaton import Automaton
from splearn.hankel import Hankel
import scipy.sparse as sps

class Serializer(object):
    """ Serializer is an helping object for data serialization
    """

    @staticmethod
    def __serialize(data):
        if data is None or isinstance(data, (bool, int, float, str)):
            if type(data).__module__ == "numpy":
                return np.asscalar(data)
            return data
        if isinstance(data, list):
            return [Serializer.__serialize(val) for val in data]
        if isinstance(data, sps.dok_matrix):
            k_str = "({0:d},{1:d})"
            return {"scipy.dok_matrix": {"shape" : Serializer.__serialize(data.shape), "dtype":  str(data.dtype),
                                         "values" : Serializer.__serialize(dict(zip([k_str.format(i,j) for (i,j) in data.keys()],
                                                                                     data.values())))}}
        if isinstance(data, dict):
            if all(isinstance(k, str) for k in data):
                return {k: Serializer.__serialize(v) for k, v in data.items()}
            return {"dict": [[Serializer.__serialize(k), Serializer.__serialize(v)] for k, v in data.items()]}
        if isinstance(data, tuple):
            return {"tuple": [Serializer.__serialize(val) for val in data]}
        if isinstance(data, set):
            return {"set": [Serializer.__serialize(val) for val in data]}
        if isinstance(data, np.ndarray):
            return {"numpy.ndarray": {
                "values": data.tolist(),
                "dtype":  str(data.dtype)}}
        if isinstance(data, Automaton):
            data_dict = {"nbL":data.nbL, "nbS":data.nbS, "initial":data.initial, "final":data.final, 
                "transitions":data.transitions, "type":data.type}
            return {"automaton" : Serializer.__serialize(data_dict)}
        if isinstance(data, Hankel):
            data_dict = {"nbL":data.nbL, "lhankel" : data.lhankel, "version" : data.version,
                         "partial" : data.partial, "sparse" : data.sparse,
                         "build_from_sample" : data.build_from_sample, "nbEx" : data.nbEx}
            return {"hankel" : Serializer.__serialize(data_dict)}
        raise TypeError("Type %s is not serializabled" % type(data))

    @staticmethod
    def __restore_json(data_str):
        if "dict" in data_str:
            return dict(data_str["dict"])
        if "tuple" in data_str:
            return tuple(data_str["tuple"])
        if "set" in data_str:
            return set(data_str["set"])
        if "scipy.dok_matrix" in data_str:
            data = data_str["scipy.dok_matrix"]
            keys = {"shape", "dtype", "values"}
            if not keys.issubset(set(data.keys())):
                raise ValueError("The input data string (" + str(data_str) +
                                  ") should contain the following keys : \"" +
                                  '\", \"'.join(keys) + "\"")
            values = Serializer.__restore_json(data["values"])
            shape = Serializer.__restore_json(data["shape"])
            dok = sps.dok_matrix(shape, dtype=data["dtype"])
            for k, val in values.items():
                k = k.replace("(","").replace(")","")
                ind1, ind2 = k.split(",")
                dok[(int(ind1), int(ind2))] = val
            return dok
        if "numpy.ndarray" in data_str:
            data = data_str["numpy.ndarray"]
            keys = {"values", "dtype"}
            if not keys.issubset(set(data.keys())):
                raise ValueError("The input data string (" + str(data_str) +
                                  ") should contain the following keys : \"" +
                                  '\", \"'.join(keys) + "\"")
            return np.array(data["values"], dtype=data["dtype"])
        if "automaton" in data_str:
            data = Serializer.__restore_json(data_str["automaton"])
            keys = {"nbL", "nbS", "initial", "final", "transitions", "type"}
            if not keys.issubset(set(data.keys())):
                raise ValueError("The input data string (" + str(data_str) +
                                 ") should contain the following keys : \"" +
                                 '\", \"'.join(keys) + "\"")
            return Automaton(nbL=data["nbL"], nbS=data["nbS"], initial=data["initial"], final=data["final"],
                         transitions=data["transitions"], type=data["type"])
        if "hankel" in data_str:
            data = Serializer.__restore_json(data_str["hankel"])
            keys = {"nbL", "lhankel", "version", "partial", "sparse", "build_from_sample", "nbEx"}
            if not keys.issubset(set(data.keys())):
                raise ValueError("The input data string (" + str(data_str) +
                                 ") should contain the following keys : \"" +
                                 '\", \"'.join(keys) + "\"")
            H = Hankel(version=data["version"], partial=data["partial"], sparse=data["sparse"],
                       lhankel = data["lhankel"])
            if data["build_from_sample"]:
                H.build_from_sample = True
                H.nbL = data["nbL"]
                H.nbEx = data["nbEx"]
            return H
            
        return data_str
    
    @staticmethod
    def __restore_yaml(data_str):
        if data_str is None or isinstance(data_str, (bool, int, float, str)):
            return data_str
        if isinstance(data_str, list):
            return [Serializer.__restore_yaml(k) for k in data_str]
        if "dict" in data_str:
            return dict(data_str["dict"])
        if "tuple" in data_str:
            return tuple(data_str["tuple"])
        if "set" in data_str:
            return set(data_str["set"])
        if "scipy.dok_matrix" in data_str:
            data = data_str["scipy.dok_matrix"]
            keys = {"shape", "dtype", "values"}
            errorMsg = "The input data string (" + str(data_str)
            errorMsg += ") should contain the following keys : \"" + '\", \"'.join(keys) + "\""
            if data is None:
                raise ValueError(errorMsg)
            if not keys.issubset(set(data.keys())):
                raise ValueError(errorMsg)
            values = Serializer.__restore_json(data["values"])
            shape = Serializer.__restore_json(data["shape"])
            dok = sps.dok_matrix(shape, dtype=data["dtype"])
            for k, val in values.items():
                k = k.replace("(","").replace(")","")
                ind1, ind2 = k.split(",")
                dok[(int(ind1), int(ind2))] = val
            return dok
        if "numpy.ndarray" in data_str:
            data = data_str["numpy.ndarray"]
            keys = {"values", "dtype"}
            errorMsg = "The input data string (" + str(data_str)
            errorMsg += ") should contain the following keys : \"" + '\", \"'.join(keys) + "\""
            if data is None:
                raise ValueError(errorMsg)
            if not keys.issubset(set(data.keys())):
                raise ValueError(errorMsg)
            return np.array(data["values"], dtype=data["dtype"])
        if "automaton" in data_str:
            data = Serializer.__restore_yaml(data_str["automaton"])
            keys = {"nbL", "nbS", "initial", "final", "transitions", "type"}
            errorMsg = "The input data string (" + str(data_str)
            errorMsg += ") should contain the following keys : \"" + '\", \"'.join(keys) + "\""
            if data is None:
                raise ValueError(errorMsg)
            if not keys.issubset(set(data.keys())):
                raise ValueError(errorMsg)
            return Automaton(nbL=data["nbL"], nbS=data["nbS"], initial=Serializer.__restore_yaml(data["initial"]),
                             final=Serializer.__restore_yaml(data["final"]),
                             transitions=[Serializer.__restore_yaml(k) for k in data["transitions"]],
                             type=data["type"])
        if "hankel" in data_str:
            data = Serializer.__restore_yaml(data_str["hankel"])
            keys = {"nbL", "lhankel", "version", "partial", "sparse", "build_from_sample", "nbEx"}
            errorMsg = "The input data string (" + str(data_str)
            errorMsg += ") should contain the following keys : \"" + '\", \"'.join(keys) + "\""
            if data is None:
                raise ValueError(errorMsg)
            if not keys.issubset(set(data.keys())):
                raise ValueError(errorMsg)
            H = Hankel(version=data["version"], partial=data["partial"], sparse=data["sparse"],
                       lhankel = [Serializer.__restore_yaml(k) for k in data["lhankel"]])
            if data["build_from_sample"]:
                H.build_from_sample = True
                H.nbL = data["nbL"]
                H.nbEx = data["nbEx"]
            return H
        return data_str
    
    @staticmethod
    def data_to_json(data):
        """ return a string into json format that does contains the input data.
        
        - Input:
        
        :param data: data composed by any types that is serializabled
        
        - Output:
        
        :returns: the output string
        :rtype: str
        """
        
        import json
        return json.dumps(Serializer.__serialize(data))

    @staticmethod
    def json_to_data(json_data_str):
        """ return a data from input json string.
        
        - Input:
        
        :param json_data_str: the json input string
        
        - Output:
        
        :returns: the data
        :rtype: deduced form the json input string
        """
        
        import json
        return json.loads(json_data_str, object_hook=Serializer.__restore_json)
    
    @staticmethod
    def data_to_yaml(data):
        """ return a string into yaml format that does contains the input data.
        
        - Input:
        
        :param data: data composed by any types that is serializabled
        
        - Output:
        
        :returns: the output string
        :rtype: str
        """
        import yaml
        return yaml.dump(Serializer.__serialize(data))

    @staticmethod
    def yaml_to_data(yaml_data_str):
        """ return a data from input yaml string.
        
        - Input:
        
        :param yaml_data_str: the yaml input string
        
        - Output:
        
        :returns: the data
        :rtype: deduced form the yaml input string
        """
        import yaml
        return Serializer.__restore_yaml(yaml.load(yaml_data_str))
