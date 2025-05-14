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
"""This module contains the Spectral and Learning class

.. module author:: François Denis

"""
from __future__ import division, print_function
import numpy as np
import math
import warnings
from concurrent.futures import ThreadPoolExecutor, wait
import threading
lock = threading.RLock()


from splearn.datasets.data_sample import SplearnArray
from splearn.hankel import Hankel
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import NotFittedError


class Spectral(BaseEstimator):
    """A Spectral estimator instance

    - Input:

    :param int rank: the ranking number
    :param lrows: (default value = 7) number or list of rows
           a list of strings or an interger indicating the max length
           of elements to consider if partial=True
           otherwise, based on self.pref if version="classic" or
           "prefix", self.fact otherwise
    :type lrows: int or tuple of int
    :param lcolumns: (default value = 7) number or list of columns
           a list of strings or an interger indicating the max length
           of elements to consider if partial=True
           otherwise, based on self.suff if version="classic" or "suffix",
           self.fact otherwise
    :type lcolumns: int or tuple of int
    :param string version: (default value = "classic") version name
    :param boolean partial: (default value = False) build
           of partial Hankel matrix
    :param boolean sparse: (default value = False) True if Hankel
           matrix is sparse
    :param boolean full_svd_calculation: (default value = False) if True the entire SVD is calculated
           for building hankel matrix. Else it is done by the sklearn random algorithm only for the greatest
           k=rank eigenvalues.
    :param string smooth_method: (default value = "none") method of smoothing

           - 'trigram' the 3-Gram trigram dict
             is computed and used by the predict function,
             in this case the threeGram probability is used instead of Spectral
             probability in negative case

           - 'none' or  something else no smooth method is used
             in predict function.
    :param boolean mode_quiet: (default value = False) True for no
           output message.

    :Example:

    >>> from splearn.spectral import Spectral
    >>> sp = Spectral()
    >>> sp.set_params(partial=True, lcolumns=6, lrows=6, smooth_method='trigram')
    Spectral(lcolumns=6, lrows=6, mode_quiet=False, partial=True, rank=5,
     smooth_method='trigram', sparse=True, version='classic')
    >>> sp.fit(data.data)
    Start Hankel matrix computation
    End of Hankel matrix computation
    Start Building Automaton from Hankel matrix
    End of Automaton computation
    Spectral(lcolumns=6, lrows=6, partial=True, rank=5, smooth_method='trigram', sparse=True, version='classic')
    >>> sp.automaton.initial
    array([-0.00049249,  0.00304676, -0.04405996, -0.10765322, -0.08660063])
    >>> sp.predict(data.data)
    array([  4.38961058e-04,   1.10616861e-01,   1.35569353e-03, ...,
        4.66041996e-06,   4.68177275e-02,   5.24287604e-20])
    >>> sp.loss(data.data, normalize=True)
    -10.530029936056017
    >>> sp.score(data.data)
    10.530029936056017

    """
    def __init__(self,  rank=5, lrows=7, lcolumns=7,
                 version='classic', partial=True,
                 sparse=True, full_svd_calculation=False, smooth_method='none',
                 mode_quiet=False):
        self.version = version
        self.partial = partial
        self.sparse = sparse
        self.full_svd_calculation = full_svd_calculation
        self.lrows = lrows
        self.lcolumns = lcolumns
        self.rank = rank
        self.trigram = {}
        self.smooth_method = smooth_method
        self._rule_smooth_method(smooth_method)
        self.mode_quiet = mode_quiet
        self._automaton = None
        self._hankel = None

    def get_params(self, deep=True):
        # suppose this estimator has parameters
        """
        return parameters values of Spectral estimator

        - Output:

        :returns: parameters dictionary of Spectral estimator name : value
        :rtype: dict

        """
        return {"rank": self.rank, "version": self.version,
                 "lrows": self.lrows, "lcolumns": self.lcolumns,
                 "partial": self.partial,
                 "sparse": self.sparse,
                 "full_svd_calculation" : self.full_svd_calculation,
                 "smooth_method" : self.smooth_method,
                 "mode_quiet" : self.mode_quiet
                }
    @property
    def automaton(self):
        """Automaton build by the fit method. None by default"""
        return self._automaton

    @automaton.setter
    def automaton(self, automaton):
        pass
    
    @property
    def hankel(self):
        """Hankel build by the fit method. None by default"""
        return self._hankel

    @hankel.setter
    def hankel(self, hankel):
        pass

    def _rule_smooth_method(self, value):
        if self.smooth_method not in ['none', 'trigram']:
            warnings.warn("smooth method should be in ['none', 'trigram']",
                          UserWarning)
            self.smooth_method = 'none'
        if value == 'trigram':
            self.smooth = 1
        else:
            self.smooth = 0

    def set_params(self, **parameters):
        """set the values of  Spectral estimator parameters

        - Output:

        :returns: Spectral estimator with new parameters
        :rtype: Spectral
        """
        for parameter, value in parameters.items():
            self.__setattr__(parameter, value)
            if parameter == "smooth_method":
                self._rule_smooth_method(value)
        return self

    def fit(self, X, y=None):
        """Fit the model

        - Input:

        :param SplearnArray X: object of shape [n_samples,n_features]
               Training data
        :param ndarray y: (default value = None) not used by Spectral estimator
               numpy array of shape [n_samples] Target values


        - Output:

        :returns: Spectral itself with an automaton attribute instanced
                  returns an instance of self.
        :rtype: Spectral

        """
        check_array(X)

        if not isinstance(X, SplearnArray):
            self._hankel = None
            self._automaton = None
            return self
        X = self.polulate_dictionnaries(X)
        self._hankel = Hankel(sample_instance=X,
                         lrows=self.lrows, lcolumns=self.lcolumns,
                         version=self.version,
                         partial=self.partial, sparse=self.sparse,
                         full_svd_calculation = self.full_svd_calculation,
                         mode_quiet=self.mode_quiet)
        self._automaton = self._hankel.to_automaton(self.rank, self.mode_quiet)
        # for smooth option compute trigram dictionnary
        if self.smooth:
            self.trigram = self._threegramdict(X.sample)

        return self

    def polulate_dictionnaries(self, X):
        """Populates the *sample*, *pref*, *suff*, *fact* dictionnaries of X
                
        - Input:

        :param SplearnArray X: object of shape [n_samples,n_features]
               Training data
        
        """
        if not isinstance(X, SplearnArray):
            return X
        dsample = {}  # dictionary (word,count)
        dpref = {}  # dictionary (prefix,count)
        dsuff = {}  # dictionary (suffix,count)
        dfact = {}  # dictionary (factor,count)
        if self.partial:
            if isinstance(self.lrows, int):
                lrowsmax = self.lrows
                version_rows_int = True
            else:
                version_rows_int = False
                lrowsmax = self.lrows.__len__()
            if isinstance(self.lcolumns, int):
                lcolumnsmax = self.lcolumns
                version_columns_int = True
            else:
                lcolumnsmax = self.lcolumns.__len__()
                version_columns_int = False
            lmax = lrowsmax + lcolumnsmax

        for line in range(X.shape[0]):
            w = X[line, :]
            w = w[w >= 0]
            w = tuple([int(x) for x in w[0:]])
            dsample[w] = dsample[w] + 1 if w in dsample else 1
            if self.version == "prefix" or self.version == "classic":
                # empty word treatment for prefixe, suffix, and factor dictionnaries
                dpref[()] = dpref[()] + 1 if () in dpref else 1
            if self.version == "suffix" or self.version == "classic":
                dsuff[()] = dsuff[()] + 1 if () in dsuff else 1
            if self.version == "factor" or self.version == "suffix" \
                    or self.version == "prefix":
                dfact[()] = dfact[()] + len(w) + 1 if () in dfact else len(
                    w) + 1
            if self.partial:
                for i in range(len(w)):
                    if self.version == "classic":
                        if (version_rows_int is True and
                                        i + 1 <= lrowsmax) or \
                        (version_rows_int is False and
                                    w[:i + 1] in self.lrows):
                            dpref[w[:i + 1]] = \
                                dpref[w[:i + 1]] + 1 if w[
                                                        :i + 1] in dpref else 1
                        if (version_columns_int is True and
                                        i + 1 <= lcolumnsmax) or \
                                (version_columns_int is False and
                                         w[-( i + 1):] in self.lcolumns):
                            dsuff[w[-(i + 1):]] = dsuff[w[-(i + 1):]] + 1 \
                                if w[-(i + 1):] in dsuff else 1
                    if self.version == "prefix":
                        # dictionaries dpref is populated until
                        # lmax = lrows + lcolumns
                        # dictionaries dfact is populated until lcolumns
                        if ((version_rows_int is True or
                                     version_columns_int is True) and
                                        i + 1 <= lmax) or \
                                (version_rows_int is False and
                                     (w[:i + 1] in self.lrows)) or \
                                (version_columns_int is False and
                                     (w[:i + 1] in self.lcolumns)):
                            dpref[w[:i + 1]] = dpref[w[:i + 1]] + 1 \
                                if w[:i + 1] in dpref else 1
                        for j in range(i + 1, len(w) + 1):
                            if (version_columns_int is True and (
                                j - i) <= lmax) or \
                                    (version_columns_int is False and
                                         (w[i:j] in self.lcolumns)):
                                dfact[w[i:j]] = dfact[w[i:j]] + 1 \
                                    if w[i:j] in dfact else 1
                    if self.version == "suffix":
                        if ((version_rows_int is True or
                                     version_columns_int is True) and
                                    i <= lmax) or \
                                (version_rows_int is False and
                                     (w[-(i + 1):] in self.lrows)) or \
                                (version_columns_int is False and
                                     (w[-(i + 1):] in self.lcolumns)):
                            dsuff[w[-(i + 1):]] = dsuff[w[-(i + 1):]] + 1 \
                                if w[-(i + 1):] in dsuff else 1
                        for j in range(i + 1, len(w) + 1):
                            if (version_rows_int is True and (
                                j - i) <= lmax) or \
                                    (version_rows_int is False and
                                         (w[i:j] in self.lrows)):
                                dfact[w[i:j]] = dfact[w[i:j]] + 1 \
                                    if w[i:j] in dfact else 1
                    if self.version == "factor":
                        for j in range(i + 1, len(w) + 1):
                            if ((version_rows_int is True or
                                         version_columns_int is True) and
                                        (j - i) <= lmax) or \
                                    (version_rows_int is False and
                                         (w[i:j] in self.lrows)) or \
                                    (version_columns_int is False and
                                         (w[i:j] in self.lcolumns)):
                                dfact[w[i:j]] = \
                                    dfact[w[i:j]] + 1 if w[i:j] in dfact else 1

            else:  # not partial
                for i in range(len(w)):
                    dpref[w[:i + 1]] = dpref[w[:i + 1]] + 1 \
                        if w[:i + 1] in dpref else 1
                    dsuff[w[i:]] = dsuff[w[i:]] + 1 if w[i:] in dsuff else 1
                    for j in range(i + 1, len(w) + 1):
                        dfact[w[i:j]] = dfact[w[i:j]] + 1 \
                            if w[i:j] in dfact else 1
        X.sample = dsample
        if self.version == "classic":
            X.pref = dpref
            X.suff = dsuff
            X.fact = {}
        if self.version == "suffix":
            X.suff = dsuff
            X.fact = dfact
            X.pref = {}
        if self.version == "prefix":
            X.pref = dpref
            X.fact = dfact
            X.suff = {}
        if self.version == "factor":
            X.fact = dfact
            X.suff = {}
            X.pref = {}
        return X

    def _populate_sample_dict(self, X):
        dsample = {}  # dictionary (word,count)
        for line in range(X.shape[0]):
            w = X[line, :]
            w = w[w >= 0]
            w = tuple([int(x) for x in w[0:]])
            dsample[w] = dsample[w] + 1 if w in dsample else 1
        return dsample

    @property
    def trigram(self):
        """The trigram dictionary"""
        return self._trigram

    @trigram.setter
    def trigram(self, DPdict_values):
        if (not isinstance(DPdict_values, dict)):
            mess = "DPdict should be a dicionary.\n"
            mess += "Actual : " + str(DPdict_values)
            raise TypeError(mess)
        self._trigram = DPdict_values

    def _trigramprobability(self, sequence, trigram_test):
        prob = np.float64(1.0)
        seq = list(sequence)
        ngramseq = [-1, -1] + seq + [-2]
        if len(seq) < 0:
            return 0
        for start in range(len(ngramseq) - 2):
            end = start + 2
            if tuple(ngramseq[start:end]) in self.trigram.keys():
                if ngramseq[end] in self.trigram[tuple(ngramseq[start:end])].keys():
                    val1_train = np.float64(
                        self.trigram[tuple(ngramseq[start:end])][ngramseq[end]])
                else:
                    val1_train = 0
                val2_train = np.float64(self.trigram[tuple(ngramseq[start:end])][-1])
            else:
                val1_train = -1
                val2_train = -1
            if tuple(ngramseq[start:end]) in trigram_test.keys():
                if ngramseq[end] in trigram_test[
                    tuple(ngramseq[start:end])].keys():
                    val1_test = np.float64(
                        trigram_test[tuple(ngramseq[start:end])][
                            ngramseq[end]])
                else:
                    val1_test = 0
                val2_test = np.float64(
                    trigram_test[tuple(ngramseq[start:end])][-1])
            else:
                val1_test = -1
                val2_test = -1
            if val1_test == -1 and val1_train == -1:
                return 0
            if val1_test == -1:
                prob = prob * val1_train / val2_train
            if val1_train == -1:
                prob = prob * val1_test / val2_test
            if val1_test != -1 and val1_train != -1:
                prob = prob * (val1_test + val1_train) / ( val2_test + val2_train)
        return prob

    def nb_trigram(self):
        """return the number of index affected by the trigram computation


        - Output:

        :returns: int number of trigram_index

        """

        try:
            nb = np.where(self.trigram_index == True)[0].shape[0]
            return nb
        except:
            warnings.warn(UserWarning, "trigram_index does not exist")
            pass


    @staticmethod
    def _threegramdict(sample):
        DPdict = dict()
        for sequence in sample.keys():
            seq = list(sequence)
            ngramseq = [-1, -1] + seq + [-2]
            for start in range(len(ngramseq) - 2):
                end = start + 2
                if tuple(ngramseq[start:end]) in DPdict:
                    table = DPdict[tuple(ngramseq[start:end])]
                    if ngramseq[end] in table:
                        table[ngramseq[end]] = table[ngramseq[end]] + sample[
                        sequence]
                    else:
                        table[ngramseq[end]] = sample[sequence]
                    table[-1] = table[-1] + sample[sequence]
                else:
                    table = dict()
                    table[ngramseq[end]] = sample[sequence]
                    table[-1] = sample[sequence]
                    DPdict[tuple(ngramseq[start:end])] = table
        return DPdict

    def predict(self, X):
        """Predict using the Spectral model

        - Input:

        :param SplearnArray X : of shape data shape = (n_samples, n_features)
               Samples.


        - Output:

        :returns: Probability corresponding to the input X,
                  array-like of shape = n_samples
        :rtype: ndarray
        """

        check_array(X)
        if not hasattr(self, 'automaton'):
            raise NotFittedError("This %(name)s instance is not fitted "
                                 "yet" % {'name': type(self).__name__})
        if self._automaton is None:
            print("No Automaton has been computed, "
                  "check the format of the input fit data")
            warnings.warn("check the format of the input fit data", UserWarning)
            return X

        Y = self.predict_proba(X)
        return Y

    def predict_proba(self, X):
        """
        Predict probability using the Spectral model

        - Input:

        :param SplearnArray X : Samples, data shape = (n_samples, n_features)


        - Output:

        :returns: Probability corresponding to the input X
                  of shape = (n_samples)
        :rtype: ndarray
        """
        #check_is_fitted(self, "classes_")
        X = check_array(X)
        if not hasattr(self, 'automaton'):
            raise NotFittedError("This %(name)s instance is not fitted "
                                 "yet" % {'name': type(self).__name__})

        # if Automaton is None because the fit pass through doing nothing
        if self._automaton is None:
            print("No Automaton has been computed, "
                  "check the format of the input fit data")
            warnings.warn("check the format of the input fit data", UserWarning)
            return X[:,0]
        # if self.smooth == 1 and self.trigram == {}:
        #     warnings.warn("Incompatibility of smooth_method "
        #                   " activate trigram smooth option in predictor "
        #                   " and fit again", UserWarning)
        #     self.trigram = self._threegramdict(X.sample)
        if self.smooth == 1:
            test_sample = self._populate_sample_dict(X=X)
            trigram_test = self._threegramdict(test_sample)
            trigram_index = np.zeros(X.shape[0], dtype=bool)
        Y = np.zeros(X.shape[0], dtype=np.float64)
        i = 0
        for line in range(X.shape[0]):
            w = X[line, :]
            w = w[w >= 0]
            w = tuple([int(x) for x in w[0:]])
            val = self._automaton.val(w)
            if self.smooth == 1 and val <= 0:
                Y[i] = self._trigramprobability(w, trigram_test)
                trigram_index[i] = True
            else:
                Y[i] = val
            i += 1
        if self.smooth == 1:
            self.trigram_index = trigram_index
        return Y

    def loss(self, X, y=None, normalize=True):
        """Log probability using the Spectral model

        - Input:

        :param SplearnArray X: of shape data shape = (n_samples, n_features)
               Samples. X is validation data.
        :param ndarray y: (default value = Null)
               numpy array of shape [n_samples] Target values,
               is the ground truth target for X (in the supervised case) or
               None (in the unsupervised case)
        :param boolean normalize: (default value = True) calculation are
               performed and normalize by the number of sample in case of True

        - Output:

        :returns: mean of Log Probability corresponding to the input X
        :rtype: float
        """
        warnings.simplefilter("error", RuntimeWarning)
        predict_prob = self.predict_proba(X)
        if y is None:
            try:
                if normalize:
                    Y = np.mean(-np.log(predict_prob))
                else:
                    Y = np.sum(-np.log(predict_prob))
            except:
                msg = "function loss or score use log " + \
                      "function, values can't be" + \
                      " negative, use it with smooth_method" + \
                      " to avoid such problem"
                raise ValueError(msg)
            return Y
        else:
            if normalize:
                Y = np.mean((np.subtract(predict_prob, y) ** 2.0))
            else:
                Y = np.sum((np.subtract(predict_prob, y) ** 2.0))
            return Y


    def score(self, X, y=None, scoring="perplexity"):
        """score of the input target

        - Input:

        :param SplearnArray X: of shape data shape = (n_samples, n_features)
               Samples.
        :param ndarray y: (default value = None)
               numpy array of shape [n_samples] Target values,
               is the ground truth target for X (in the supervised case) or
               None (in the unsupervised case)
        :param string scoring: (default value = "perplexity")
               method for score computation

        - Output:

        :returns: score, on the input X
        :rtype: float
        """


        if scoring == "perplexity":
            if y is None:
                return - self.loss(X, y, normalize=True)
            else:
                predict_prob = self.predict_proba(X)
                sA, sC = 0, 0
                sA = sum(predict_prob)
                sC = sum(y)
                s = 0
                perplexity = 0
                for i in range(X.shape[0]):
                    try:
                        s = s + y[i] / sC * math.log(predict_prob[i] / sA)
                        perplexity = math.exp(-s)
                    except:
                        msg = "function loss or score use log " + \
                              "function values can't be" + \
                              " negative, use it with smooth_method" + \
                              "to avoid such problem"
                        raise ValueError(msg)
                return perplexity
        else:
            return - self.loss(X, y, normalize=True)