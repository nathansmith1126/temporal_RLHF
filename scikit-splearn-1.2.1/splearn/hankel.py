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
"""This module contains the Hankel class

"""
from __future__ import division, print_function
import scipy.sparse as sps
import scipy.sparse.linalg as lin
import numpy as np
from numpy.linalg import svd, pinv
from sklearn.utils.extmath import randomized_svd as sk_svd

class Hankel(object):
    """ A Hankel instance , compute the list of Hankel matrices

    - Input:
    
    :param SplearnArray sample_instance: instance of SplearnArray
    :param lrows: number or list of rows,
           a list of strings if partial=True;
           otherwise, based on self.pref if version="classic" or
           "prefix", self.fact otherwise
    :type lrows: int or list of int
    :param lcolumns: number or list of columns
           a list of strings if partial=True ;
           otherwise, based on self.suff if version="classic" or "suffix",
           self.fact otherwise
    :type lcolumns: int or list of int
    :param string version: (default = "classic") version name
    :param boolean partial: (default value = False) build of partial
    :param boolean sparse: (default value = False) True if Hankel
           matrix is sparse
    :param boolean full_svd_calculation: (default value = False) if True the entire SVD is calculated
           for building hankel matrix. Else it is done by the sklearn random algorithm only for the greatest
           k=rank eigenvalues.
    :param boolean mode_quiet: (default value = False) True for no
           output message.
    :param list lhankel: list of all Hankel matrices. At least one of the two parameters 
           *sample_instance* or *lhankel* has to be not None. If *sample_instance* is given,
           the **Hankel** instance is built directly from the sample dictionnary,
           else it is deduced from the *lhankels* list of matrices. 

    :Example:
    
    >>> from splearn import Learning, Hankel , Spectral
    >>> train_file = '0.spice.train'
    >>> pT = load_data_sample(adr=train_file)
    >>> sp = Spectral()
    >>> sp.fit(X=pT.data)
    >>> lhankel = Hankel( sample_instance=pT.sample,
    >>>                   nbL=pT.nbL, nbEx=pT.nbEx,
    >>>                   lrows=6, lcolumns=6, version="classic",
    >>>                   partial=True, sparse=True, mode_quiet=True).lhankel

    """

    def __init__(
            self, sample_instance=None,
            lrows=[], lcolumns=[],
            version="classic", partial=False,
            sparse=False, full_svd_calculation=False, mode_quiet=False, lhankel=None):
        
        self.version = version
        self.partial = partial
        self.sparse = sparse
        self.full_svd__calculation = full_svd_calculation
        self.build_from_sample = True
        if sample_instance is not None:
            # Size of the alphabet
            self.nbL = sample_instance.nbL
            # Number of samples
            self.nbEx = sample_instance.nbEx
            self.lhankel = self.build(sample=sample_instance.sample,
                                      pref=sample_instance.pref,
                                      suff=sample_instance.suff,
                                      fact=sample_instance.fact,
                                      lrows=lrows, lcolumns=lcolumns,
                                      mode_quiet=mode_quiet)
        elif lhankel is not None:
            # Size of the alphabet
            self.nbL = len(lhankel) - 1
            self.lhankel = lhankel
            self.build_from_sample = False
            self._nbEx = -1
        else:
            raise ValueError("At least sample_instance or lhankel has to be not None.")

    def __eq__(self, other):
        #print("Hankel equality check")
        if self.version != other.version:
        #    print("version is different")
            return False
        if self.partial != other.partial:
        #    print("partial is different")
            return False
        if self.sparse != other.sparse:
        #    print("sparse is different")
            return False
        if self.build_from_sample != other.build_from_sample:
        #    print("build_from_sample is different")
            return False
        if self.nbL != other.nbL:
        #    print("nbL is different")
            return False
        if self.nbEx != other.nbEx:
        #    print("nbEx is different")
            return False
        if len(self.lhankel) != len(other.lhankel):
        #    print("lhankel length is different")
            return False
        for lh1, lh2 in zip(self.lhankel, other.lhankel):
            if self.sparse:
                if (lh1 != lh2).nnz > 0:
                #    print("{:d} elements oh lhandel are different".format((lh1 != lh2).nnz))
                    return False
            elif not np.array_equal(lh1, lh2):
            #    print("Different Array")
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def nbL(self):
        """Number of letters"""
        return self._nbL

    @nbL.setter
    def nbL(self, nbL):
        if not isinstance(nbL, int):
            raise TypeError("nbL should be an integer")
        if nbL < 0:
            raise ValueError("The size of the alphabet should " +
                             "an integer >= 0")
        self._nbL = nbL

    @property
    def nbEx(self):
        """Number of examples"""
        return self._nbEx

    @nbEx.setter
    def nbEx(self, nbEx):
        if not isinstance(nbEx, int):
            raise TypeError("nbEx should be an integer")
        if nbEx < 0:
            raise ValueError("The number of examples should be " +
                             " an integer >= 0")
        self._nbEx = nbEx
    
    @property
    def build_from_sample(self):
        """Boolean that indicates if the matrices have been build form sample or not
        (directly build from an Automaton in this case) """
        return self._build_from_sample
    
    @build_from_sample.setter
    def build_from_sample(self, val):
        if val:
            self._build_from_sample = True
        else:
            self._build_from_sample = False
        

    def build(self, sample, pref, suff, fact, lrows, lcolumns, mode_quiet):

        """ Create a Hankel matrix

        - Input:

        :param dict sample: the keys are the words and the values are the number of time it appears in the sample.
        :param dict pref: the keys are the prefixes and the values are the number of time it appears in the sample.
        :param dict suff: the keys are the suffixes and the values are the number of time it appears in the sample.
        :param dict fact: the keys are the factors and the values are the number of time it appears in the sample.
        :param lrows: number or list of rows,
               a list of strings if partial=True;
               otherwise, based on self.pref if version="classic" or
               "prefix", self.fact otherwise
        :type lrows: int or list of int
        :param lcolumns: number or list of columns
               a list of strings if partial=True ;
               otherwise, based on self.suff if version="classic" or "suffix",
               self.fact otherwise
        :type lcolumns: int or list of int
        :param boolean mode_quiet:  True for no output message.

        - Output:

        :returns: list lhankel, list of hankel matrix,
                  a DoK based sparse matrix or nuppy matrix based not sparse
        :rtype: list of matrix

        """
        # calcul des lignes lrows et colonnes lcolumns
        if not mode_quiet:
            print("Start Hankel matrix computation")
        if not self.partial:
            (lrows, lcolumns) = self._build_not_partial(
                pref=pref, suff=suff, fact=fact)
        else:
            (lrows, lcolumns) = self._build_partial(
                pref=pref, suff=suff, fact=fact,
                lrows=lrows, lcolumns=lcolumns)

        lhankel = self._create_hankel(sample=sample, pref=pref,
                                      suff=suff, fact=fact,
                                      lrows=lrows, lcolumns=lcolumns)
        if not mode_quiet:
            print ("End of Hankel matrix computation")
        return lhankel

    def _build_not_partial(self,pref, suff, fact):
        version = self.version
        if version == "classic":
            lrows = pref.keys()
            lcolumns = suff.keys()
        elif version == "prefix":
            lrows = pref.keys()
            lcolumns = fact.keys()
        elif version == "suffix":
            lrows = fact.keys()
            lcolumns = suff.keys()
        else:
            lrows = fact.keys()
            lcolumns = fact.keys()

        return (lrows, lcolumns)

    def _build_partial(self,
                       pref, suff, fact,
                       lrows, lcolumns):
        version = self.version

        if version == "classic":
            (lrows, lcolumns) = self._construc_partial_lrows_lcolumns(
                dict_first=pref, dict_second=suff,
                lrows=lrows, lcolumns=lcolumns)
        elif version == "prefix":
            (lrows, lcolumns) = self._construc_partial_lrows_lcolumns(
                dict_first=pref, dict_second=fact,
                lrows=lrows, lcolumns=lcolumns)
        elif version == "suffix":
            (lrows, lcolumns) = self._construc_partial_lrows_lcolumns(
                dict_first=fact, dict_second=suff,
                lrows=lrows, lcolumns=lcolumns)
        else:
            (lrows, lcolumns) = self._construc_partial_lrows_lcolumns(
                dict_first=fact, dict_second=fact,
                lrows=lrows, lcolumns=lcolumns)
        return lrows, lcolumns

    def _construc_partial_lrows_lcolumns(self, dict_first, dict_second,
                                         lrows,
                                         lcolumns):

        if isinstance(lrows, int):
            longmax = lrows
            lrows = [w for w in dict_first if len(w) <= longmax]
        else:
            s_first = set(dict_first)  # corresponding set
            lrows = [w for w in lrows if w in s_first]
        if isinstance(lcolumns, int):
            longmax = lcolumns
            lcolumns = [w for w in dict_second if len(w) <= longmax]
        else:
            s_second = set(dict_second)  # corresponding set
            lcolumns = [w for w in lcolumns if w in s_second]
        return (lrows, lcolumns)

    def _create_hankel(self, sample, pref, suff, fact, lrows, lcolumns):
        version = self.version
        sparse = self.sparse

        (drows, dcolumns) = self._sorted_rows_columns(lrows, lcolumns)

        nbRows = len(lrows)
        nbColumns = len(lcolumns)
        srows = set(lrows)
        scolumns = set(lcolumns)

        if sparse:
            lhankel = [sps.dok_matrix((nbRows, nbColumns)) for
                       i in range(self.nbL+1)]
        else:
            lhankel = [np.zeros((nbRows, nbColumns), dtype=np.float64) for
                       k in range(self.nbL+1)]
        if version == "classic":
            dsample = sample
        elif version == "prefix":
            dsample = pref
        elif version == "suffix":
            dsample = suff
        else:
            dsample = fact
        for w in dsample:
            for i in range(len(w)+1):
                if w[:i] in srows:
                    if w[i:] in scolumns:
                        lhankel[0][drows[w[:i]], dcolumns[w[i:]]] = dsample[w]
                    if (i < len(w) and w[i+1:] in scolumns):
                        lhankel[w[i]+1][drows[w[:i]],
                                        dcolumns[w[i+1:]]] = dsample[w]
        return lhankel

    def _sorted_rows_columns(self, lrows, lcolumns):
        nbRows = len(lrows)
        nbColumns = len(lcolumns)
        lrows = sorted(lrows, key=lambda x: (len(x), x))
        drows = {lrows[i]: i for i in range(nbRows)}
        lcolumns = sorted(lcolumns, key=lambda x: (len(x), x))
        dcolumns = {lcolumns[i]: i for i in range(nbColumns)}

        return (drows, dcolumns)
    
    def to_automaton(self, rank, mode_quiet=False):
        """ Return an automaton from the current Hankel matrix

        - Input:

        :param int rank: the matrix rank
        :param boolean mode_quiet: True for no output message.

        - Output:

        :returns: An automaton instance
        :rtype: Automaton
        """
        from splearn.automaton import Automaton
        if not mode_quiet:
            print("Start Building Automaton from Hankel matrix")
        matrix_shape =min(self.lhankel[0].shape)
        if (min(self.lhankel[0].shape) < rank) :
            raise ValueError("The value of parameter rank ("+  str(rank)
                             + ") should be less than " +
                              "the smaller dimension of the Hankel Matrix (" +
                             str(matrix_shape) + ")")
        if not self.sparse:
            hankel = self.lhankel[0]
            if self.full_svd__calculation:
                [u, s, v] = svd(hankel)
                u = u[:, :rank]
                v = v[:rank, :]
                # ds = np.zeros((rank, rank), dtype=complex)
                ds = np.diag(s[:rank]).astype(np.float64)
            else:
                [u, s, v] = sk_svd(hankel, n_components=rank)
                ds = np.diag(s).astype(np.float64)
            pis = pinv(v)
            del v
            pip = pinv(np.dot(u, ds))
            del u, ds
            init = np.dot(hankel[0, :], pis)
            term = np.dot(pip, hankel[:, 0])
            trans = []
            for x in range(self.nbL):
                hankel = self.lhankel[x+1]
                trans.append(np.dot(pip, np.dot(hankel, pis)))

        else:
            hankel = self.lhankel[0].tocsr()
            if self.full_svd__calculation:
                [u, s, v] = svd(hankel.A)
                u = u[:, :rank]
                v = v[:rank, :]
                # ds = np.zeros((rank, rank), dtype=complex)
                ds = np.diag(s[:rank]).astype(np.float64)
            else:
                [u, s, v] = lin.svds(hankel, k=rank)
                ds = np.diag(s).astype(np.float64)
            pis = pinv(v)
            del v
            pip = pinv(np.dot(u, ds))
            del u, ds
            init = hankel[0, :].dot(pis)[0, :]
            term = np.dot(pip, hankel[:, 0].toarray())[:, 0]
            trans = []
            for x in range(self.nbL):
                hankel = self.lhankel[x+1].tocsr()
                trans.append(np.dot(pip, hankel.dot(pis)))

        A = Automaton(nbL=self.nbL, nbS=rank, initial=init, final=term,
                         transitions=trans, type=self.version)
        if self.build_from_sample:
            A.initial = A.initial / self.nbEx
            if self.version == "prefix":
                A = A.transformation(source="prefix", target="classic")
            if self.version == "factor":
                A = A.transformation(source="factor", target="classic")
            if self.version == "suffix":
                A = A.transformation(source="suffix", target="classic")
            if not mode_quiet:
                print ("End of Automaton computation")
        return A

    @staticmethod
    def write(hankel_in, filename, format='json'):
        """ write input hankel into a file with the given format.

        - Input:

        :param Hankel hankel_in: hankel to write into the file
        :param str filename: the name of the file. If it does not exist,
         the file is created.
        :param str format: 'json' or yaml'
        """
        from splearn.serializer import Serializer
        if format == 'json':
            data_str = Serializer.data_to_json(hankel_in)
        elif format == 'yaml':
            data_str = Serializer.data_to_yaml(hankel_in)
        else:
            raise ValueError("Invalid input format. Should be \"json\" or \"yaml\"")
        with open(filename, 'w') as outfile:
            outfile.write(data_str)

    @staticmethod
    def read(filename, format='json'):
        """ return a Hankel build with attributes read from a file

        - Input:

        :param str filename: the name of the input file.
        :param str format: 'json' or yaml'

        - Output:

        :returns: the output hankel
        :rtype: Hankel
        """
        from splearn.serializer import Serializer
        with open(filename) as infile:
            datastr = infile.read()
        if format == 'json':
            return Serializer.json_to_data(datastr)
        if format == 'yaml':
            return Serializer.yaml_to_data(datastr)
        raise ValueError("Invalid input format. Should be \"json\" or \"yaml\"")
