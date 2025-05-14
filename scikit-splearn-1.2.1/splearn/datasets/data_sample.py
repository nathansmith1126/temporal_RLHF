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
"""This module contains the DataSample class and SplearnArray class.


"""
import numpy as np


class SplearnArray(np.ndarray):
    """Sample data array used by the splearn spectral estimation

    **SplearnArray** class inherit from numpy ndarray as a 2d data ndarray.
    
    Example of a possible 2d shape:
    
    +---+---+---+---+---+
    |  0|  1|  0|  3| -1|
    +---+---+---+---+---+
    |  0|  0|  3|  3|  1|
    +---+---+---+---+---+
    |  1|  1| -1| -1| -1|
    +---+---+---+---+---+
    |  5| -1| -1| -1| -1|
    +---+---+---+---+---+
    | -1| -1| -1| -1| -1|
    +---+---+---+---+---+
    
    is equivalent to:
    
    - word (0103) or abad
    - word (00331) or aaddb
    - word (11) or bb
    - word (5) or f
    - word () or empty
    
    Each line represents a word of the sample. The words are represented by integer letters (0->a, 1->b, 2->c ...).
    -1 indicates the end of the word. The number of rows is the total number of words in the sample (=nbEx) and the number of columns
    is given by the size of the longest word. Notice that the total number of words does not care about the words' duplications. 
    If a word is duplicated in the sample, it is counted twice as two different examples. 
    
    The DataSample class encapsulates also the sample's parameters 'nbL', 'nbEx' (number of letters in the alphabet and 
    number of samples) and the fourth dictionaries 'sample', 'prefix', 'suffix' and 'factor' that will be populated during the fit
    estimations.
    
    - Input:

    :param nd.array input_array: input ndarray that will be converted into **SplearnArray**
    :param int nbL: the number of letters
    :param int nbEx: total number of examples.
    :param dict sample: the keys are the words and the values are the number of time it appears in the sample.
    :param dict pref: the keys are the prefixes and the values are the number of time it appears in the sample.
    :param dict suff: the keys are the suffixes and the values are the number of time it appears in the sample.
    :param dict fact: the keys are the factors and the values are the number of time it appears in the sample.

    :Example:

    >>> from splearn.datasets.base import load_data_sample
    >>> from splearn.tests.datasets.get_dataset_path import get_dataset_path
    >>> train_file = '3.pautomac_light.train' # '4.spice.train'
    >>> data = load_data_sample(adr=get_dataset_path(train_file))
    >>> print(data.__class__)
    >>> data.data
    <class 'splearn.datasets.data_sample.DataSample'>
    SplearnArray([[ 3.,  0.,  3., ..., -1., -1., -1.],
        [ 3.,  3., -1., ..., -1., -1., -1.],
        [ 3.,  2.,  0., ..., -1., -1., -1.],
        ...,
        [ 3.,  1.,  3., ..., -1., -1., -1.],
        [ 3.,  0.,  3., ..., -1., -1., -1.],
        [ 3.,  3.,  1., ..., -1., -1., -1.]])
    """
    def __new__(cls, input_array, nbL=None, nbEx=None,
                sample=None, pref=None,
                suff=None, fact=None, *args, **kwargs):
        obj = np.asarray(input_array).view(cls)
        obj.nbL = nbL
        obj.nbEx = nbEx
        obj.sample = sample
        obj.pref = pref
        obj.suff = suff
        obj.fact = fact
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.nbL = getattr(obj, 'nbL', None)
        self.nbEx = getattr(obj, 'nbEx', None)

        self.sample = getattr(obj, 'sample', None)
        self.pref = getattr(obj, 'pref', None)
        self.suff = getattr(obj, 'suff', None)
        self.fact = getattr(obj, 'fact', None)

class DataSample(dict):
    """ A DataSample instance

    - Input:

    :param tuple data: a tuple of (int, int, numpy.array) for the corresponding three elements
        (nbL, nbEx, data) where nbL is the number of letters in the alphabet, nbEx is the number
        of samples and data is the 2d data array

    :Example:

    >>> from splearn.datasets.base import load_data_sample
    >>> from splearn.tests.datasets.get_dataset_path import get_dataset_path
    >>> train_file = '3.pautomac_light.train' # '4.spice.train'
    >>> data = load_data_sample(adr=get_dataset_path(train_file))
    >>> print(data.__class__)
    <class 'splearn.datasets.data_sample.DataSample'>
    >>> data.nbL
    4
    >>> data.nbEx
    5000
    >>> data.data

    """

    def __init__(self, data=None, **kwargs):
        # The dictionary that contains the sample
        self._data = SplearnArray(np.zeros((0,0)))
        if data is not None:
            self.data = SplearnArray(data[2], nbL=data[0], nbEx=data[1])
        super(DataSample, self).__init__(kwargs)


    @property
    def nbL(self):
        """Number of letters"""
        return self.data.nbL

    @nbL.setter
    def nbL(self, nbL):
        if not isinstance(nbL, int):
            raise TypeError("nbL should be an integer")
        if nbL < 0:
            raise ValueError("The size of the alphabet should " +
                             "an integer >= 0")
        self.data.nbL = nbL

    @property
    def nbEx(self):
        """Number of examples"""
        return self.data.nbEx

    @nbEx.setter
    def nbEx(self, nbEx):
        if not isinstance(nbEx, int):
            raise TypeError("nbEx should be an integer")
        if nbEx < 0:
            raise ValueError("The number of examples should be " +
                             " an integer >= 0")
        self.data.nbEx = nbEx

    @property
    def data(self):
        """SplearnArray"""
        return self._data

    @data.setter
    def data(self, data):
        if isinstance(data, (SplearnArray, np.ndarray, np.generic)):
            self._data = data
        else:
            raise TypeError("sample should be a SplearnArray.")

