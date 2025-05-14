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
"""This module contains the Automaton class
"""

import numpy as np


class Automaton(object):
    """ Define an automaton with parameters

    - Input:

    :param int nbL: the number of letters
    :param int nbS: the number of states
    :param list initial: the initial vector
    :param list final: the final vector
    :param list transition: the transitions tables


    """

    def __init__(self, nbL=0, nbS=0, initial=[], final=[],
                 transitions=[], type='classic'):
        # The number of letters
        self.nbL = nbL
        # The number of states
        self.nbS = nbS
        # The vector containing the initial weight of each state
        self.initial = initial
        # The vector containing the final weight of each state
        self.final = final
        # The list of arrays defining the transitions
        self.transitions = transitions
        # The type of automaton
        self.type = type
        # Flag if the absolute convergence of the automaton has been calculated
        # or not
        self.__convcalculated = False
        self.__isAbsConv = False

    @property
    def final(self):
        """The vector containing the final weight of each state"""
        return self.__final

    @final.setter
    def final(self, final_values):
        if (not isinstance(final_values, np.ndarray) or
           final_values.dtype.type != np.float64):
            mess = "final_values should be a numpy.array of float.\n"
            mess += "Actual : " + str(final_values)
            raise TypeError(mess)
        if len(final_values) != self.nbS:
            raise ValueError("final_values length should be " + str(self.nbS))
        self.__final = final_values
        self.__convcalculated = False

    @property
    def initial(self):
        """The vector containing the initial weight of each state"""
        return self.__initial

    @initial.setter
    def initial(self, initial_values):
        if (not isinstance(initial_values, np.ndarray) or
           initial_values.dtype.type != np.float64):
            raise TypeError("initial_values should be a numpy.array of float" +
                            str(initial_values))
        if len(initial_values) != self.nbS:
            raise ValueError("initial_values length should be " +
                             str(self.nbS))
        self.__initial = initial_values
        self.__convcalculated = False

    @property
    def transitions(self):
        """The list of arrays defining the transitions"""
        return self.__transitions

    @transitions.setter
    def transitions(self, transitions_values):
        if not isinstance(transitions_values, list):
            raise TypeError("transitions_values should be a list")
        if len(transitions_values) != self.nbL:
            raise ValueError("The length of transitions_values should be " +
                             str(self.nbL))
        for x in transitions_values:
            if not isinstance(x, np.ndarray) or x.dtype != np.float64:
                raise TypeError("transitions_values should contain" +
                                " numpy.array of float")
            if (len(x.shape) != 2 or x.shape[0] != x.shape[1] or
                    x.shape[0] != self.nbS):
                mess = "Elements of transitions_value should "
                mess += "be {0:d}x{0:d} square matrices."
                mess = mess.format(self.nbS)
                raise ValueError(mess)
        self.__transitions = transitions_values
        self.__convcalculated = False

    @property
    def nbS(self):
        """The number of states"""
        return self.__nbS

    @nbS.setter
    def nbS(self, nbS_value):
        if (nbS_value == int(nbS_value) and nbS_value > 0):
            self.__nbS = int(nbS_value)
            self.__convcalculated = False
        else:
            raise ValueError("Error in new nbS value.")

    @property
    def nbL(self):
        """The number of letters"""
        return self.__nbL

    @nbL.setter
    def nbL(self, nbL_value):
        if (nbL_value == int(nbL_value) and nbL_value > 0):
            self.__nbL = int(nbL_value)
            self.__convcalculated = False
        else:
            raise ValueError("Error in new nbL value.")

    @property
    def isAbsConv(self):
        """Does the automaton meet the sufficient condition to be absolutely convergent"""
        if not self.__convcalculated:
            self._calcAbsConv()
            self.__convcalculated = True
        return self.__isAbsConv

    @isAbsConv.setter
    def isAbsConv(self, isAbsConvL_value):
        pass


    @property
    def type(self):
        """The string indicates the type of automaton"""
        return self._type

    @type.setter
    def type(self, type_value):
        if (not isinstance(type_value, str)):
            mess = "type_value should be a string.\n"
            mess += "Actual : " + str(type_value)
            raise TypeError(mess)
        if (type_value == 'classic' or  type_value == 'prefix' or
                    type_value == 'suffix' or type_value == 'factor'):
            self._type = type_value
        else:
            raise ValueError("type must be classic, prefix, suffix or factor.")

    def __rmul__(self, coeff):
        """ Multiplication of an automaton by a scalar

        - Input:

        :param scalar coeff: the coefficient of multiplication

        - Output:

        :returns: An automaton instance
        :rtype: Automaton

        """
        initial = coeff*self.initial
        return(Automaton(self.nbL, self.nbS, initial, self.final,
                         self.transitions))

    def __add__(self, automaton):
        """ Computes self + automaton
        self and automaton are built on the same alphabet

        - Input:

        :param Automaton automaton: automaton to add

        - Output:

        :returns: An automaton instance
        :rtype: Automaton

        """
        if self.nbL != automaton.nbL:
            raise ValueError("Can't add Automaton with different alphabets")
        else:
            initial = np.concatenate((self.initial, automaton.initial))
            final = np.concatenate((self.final, automaton.final))
            nbS = self.nbS + automaton.nbS
            nbL = self.nbL
            transitions = [np.zeros((nbS, nbS)) for _ in range(nbL)]
            for i in range(len(transitions)):
                transitions[i][0:self.nbS, 0:self.nbS] =\
                    self.transitions[i][0:self.nbS, 0:self.nbS]
                transitions[i][self.nbS:nbS, self.nbS:nbS] =\
                    automaton.transitions[i][0:automaton.nbS, 0:automaton.nbS]

        return Automaton(nbL, nbS, initial, final, transitions)

    @staticmethod
    def SimpleExample():
        """ A Probabilistic Automaton with two states and two letters.

        - Output:

        :returns: An automaton instance example with simple values
        :rtype: Automaton
        """
        initial = np.array([0.5, 0.5])
        final = np.array([1/2, 1/6])
        transitions = [np.array([[1/6, 1/12], [1/6, 1/6]]),
                       np.array([[0, 1/4], [1/6, 1/3]])]
        return Automaton(2, 2, initial, final, transitions)

    def transformation(self, source="classic", target="prefix"):
        """ Takes an automaton as input and transforms it.

        - Input:

        :param str source: "prefix", "factor" or "classic" or "suffix"(default)
        :param str target: "prefix" (default) "factor" or "classic" or "suffix"

        - Output:

        :returns: The result automaton instance
        :rtype: Automaton

         The transformation is done according to the source and target
         parameters.
         .. warning:: it does not check the convergence
         """
        A = Automaton(nbL=self.nbL, nbS=self.nbS, initial=self.initial,
                      final=self.final,
                      transitions=self.transitions)
        m_sigma = np.zeros(self.nbS)
        for m in self.transitions:
            m_sigma = m_sigma + m
        m = np.eye(self.nbS) - m_sigma
        im = np.linalg.inv(m)
        if source == "classic":
            if target == "prefix" or target == "factor":
                A.final = np.dot(im, A.final)
            if target == "factor" or target == "suffix":
                A.initial = np.dot(A.initial, im)
        elif target == "classic" :
            if source == "prefix" or source == "factor":
                A.final = np.dot(m, A.final)
            if source == "factor" or source == "suffix":
                A.initial = np.dot(A.initial, m)
        A.type = target
        return A

    def BuildHankels(self, lrows=[], lcolumns=[]):
        """ Return all Hankel (denses) matrices built on lrows and lcolumns from an automaton

        - Input:

        :param list lrows:
        :param list lcolumns:

        - Output:

        :returns: list of all Hankel matrices built on lrows and lcolumns
        :rtype: list
        """
        nbrows = len(lrows)
        nbcolumns = len(lcolumns)
        lh = [np.zeros((nbrows, nbcolumns)) for _ in range(self.nbL + 1)]
        dval = {}  # dictionary of already calculated values
        for i in range(nbrows):
            for j in range(nbcolumns):
                w = lrows[i] + lcolumns[j]
                if w in dval:
                    lh[0][i, j] = dval[w]
                else:
                    lh[0][i, j] = self.val(w)
                    dval[w] = lh[0][i, j]
                for x in range(self.nbL):
                    w = lrows[i] + (x,) + lcolumns[j]
                    if w in dval:
                        lh[x+1][i, j] = dval[w]
                    else:
                        lh[x+1][i, j] = self.val(w)
                        dval[w] = lh[x+1][i, j]
        return lh
    
    def to_hankel(self, lrows, lcolumns, mode_quiet=False):
        """ Return an Hankel instance (denses, classic and not partial) with matrices
        built on lrows and lcolumns from an automaton

        - Input:

        :param list lrows: prefixes
        :param list lcolumns: suffixes
        :param boolean mode_quiet: (default value = False) True for no
               output message.

        - Output:

        :returns: Hankel instance
        :rtype: Hankel
        """
        from splearn.hankel import Hankel
        lhankel = self.BuildHankels(lrows, lcolumns)
        return Hankel(mode_quiet=mode_quiet, lhankel=lhankel)

    def mirror(self):
        """ Compute the mirror automaton

        - Input:

        :param Automaton self: Automaton(nbL, nbS, initial, final, transitions)

        - Output:

        :returns: mA = Automaton(nbL, nbS, final, initial, Newtransitions)
                where Newtransitions[x] = transpose(transitions[x])
        :rtype: Automaton

        """
        Newtransitions = []
        for x in range(self.nbL):
            Newtransitions.append(np.transpose(self.transitions[x]))
        return Automaton(self.nbL, self.nbS, self.final, self.initial,
                         Newtransitions)

    def val(self, word):
        """ Compute the value computed by the automaton on word

        - Input:

        :param Automaton self: weighted automaton
        :param str word: a string

        - Output:

        :returns: probability r_A(w)
        :rtype: float

        """
        u = self.initial
        final = self.final
        for x in word:
            u = np.dot(u, self.transitions[x])
        return np.dot(u, final)

    @staticmethod
    def HouseholderReflector(x):
        """ the vector which defines the Householder for x

        - Input:

        :param vector x: a vector in :math:`R^k` different from 0

        - Output:

        :returns: :math:`v = u/||u||` 
            where :math:`u_1 = x_1 + sign(x_1)||x||`
            and :math:`u_i = x_i` for :math:`i \\geq 2`
        :rtype: vector

        """

        u = x.copy()
        s = 1 if x[0] >= 0 else -1
        u[0] = x[0] + s * np.linalg.norm(x)
        return u/np.linalg.norm(u)

    @staticmethod
    def mulHouseholderReflector(u, v):
        """ the product of u by the HouseholderReflector nxn matrix based on v

        - Input:

        :param vector u: row vector of :math:`R^n`
        :param vector v: vector of :math:`R^k` (k<=n)

        - Output:

        :returns: w, row vector of :math:`R^n`,
             :math:`w = uP(v)` where
             :math:`P(v)=[I_{n-k} 0; 0 R]\in R^{n \\times n}` and
             :math:`R=I_k-2v^T.v`
        :rtype: vector

        """
        n = len(u)
        w = u.copy()
        k = len(v)
        r = np.dot(u[n-k:n], v)
        w[n-k:n] = w[n-k:n] - 2*r*v
        return w

    def HouseholderReductionFw(self, tau):
        """ algorithm (Fig. 3) from the paper
        Stability and complexity of Minimising Probabilistic Automata
        by Kiefer and Wachter

        - Input:

        :param Automaton self: an object of the automaton class
        :param float tau: error tolerance parameter >=0

        - Output:

        :returns: The canonical forward reduction computed to the tolerance tau
        :rtype: Automaton

        """

        alpha = self.initial  # initial vector
        n = len(alpha)  # initial dimension
        v = Automaton.HouseholderReflector(alpha/np.linalg.norm(alpha))
        # vP1\in R^n
        lv = [v]  # list of Householder reflectors
        l = 0
        j = 1  # new number of dimensions
        e = np.zeros(n)
        e[0] = 1
        f = Automaton.mulHouseholderReflector(e, v)  # f \in R^n
        F = np.zeros([n, n])  # the projection matrix
        F[j-1, :] = f
        transNew = [np.zeros([n, n]) for x in range(self.nbL)]
        # new transitions
        while l < j:
            l += 1
            for x in range(self.nbL):
                f = F[l-1, :]
                u = np.dot(f, self.transitions[x])  # f_lM(a) u\in R^n
                for v in lv:
                    u = Automaton.mulHouseholderReflector(u, v)  # u \in R^n
                transNew[x][l-1, :] = u
                if (j+1 <= n and np.linalg.norm(transNew[x][l-1][j:n]) > tau):
                    j += 1
                    v = Automaton.HouseholderReflector(transNew[x][l-1][j-1:n])
                    # v \in R^{n-j}
                    lv.append(v)
                    transNew[x][l-1, :] = Automaton.mulHouseholderReflector(
                                          transNew[x][l-1, :], v)
                    f = np.zeros(n)
                    f[j-1] = 1
                    for v in reversed(lv):
                        f = Automaton.mulHouseholderReflector(f, v)
                    F[j-1, :] = f
        F = F[:j, :]
        for x in range(self.nbL):
            transNew[x] = transNew[x][:j, :j]
        iF = np.linalg.pinv(F)
        alphaNew = np.dot(alpha, iF)
        etaNew = np.dot(F, self.final)

        return Automaton(self.nbL, j, alphaNew, etaNew, transNew)

    def minimisation(self, tau):
        """ compute an equivalent minimal automaton, to the precision tau

        - Input:

        :param Automaton self:

        - Output:

        :returns: B, equivalent to A with a minimal number of states
        :rtype: Automaton

        """

        A = self.HouseholderReductionFw(tau)
        A = A.mirror()
        A = A.HouseholderReductionFw(tau)
        return A.mirror()

    def _calcAbsConv(self):
        """ a sufficient condition to be absolutely convergent

        - Input:

        :param Automaton self:

        - Output:

        :returns: False if  :math:`\\Sigma r_|A|(w)` is not convergent and
                True otherwise.
                It may happen that A is abs conv and that the return value
                is False
        :rtype: bool

        """
        m = np.zeros([self.nbS, self.nbS])
        for x in range(self.nbL):
            m = m + abs(self.transitions[x])
        if max(abs(np.linalg.eigvals(m))) < 1:
            self.__isAbsConv = True
        else:
            self.__isAbsConv = False

    def sum(self):
        """ the sum of a rational series

        - Input:

        :param Automaton self:

        - Output:

        :returns: sum over all samples of transitions
        :rtype: ndarray
        """
        m = np.zeros([self.nbS, self.nbS])
        for x in range(self.nbL):
            m = m + self.transitions[x]
        return np.dot(np.dot(self.initial, np.linalg.inv(np.eye(self.nbS)-m)),
                      self.final)

    @staticmethod
    def load_Pautomac_Automaton(adr):
        """ Load an automaton from a PAutomaC file and returns an object of the
        class Automaton; works for PFA and PDFA - not for HMM.

        - Input:

        :param string adr: address and name of the loaden file

        - Output

        :returns: An automaton instance
        :rtype: Automaton
        
        """
        states = set()  # set of states
        letters = set()  # alphabet
        dinit = {}  # dictionary of (initial state, initial value)
        dfinal = {}  # dictionary of (final state, final value)
        trans1 = {}  # [(state,letter), total weight] ; sum_l w(q,l) = 1
        trans2 = {}  # [(state,letter,state) weight] ; sum_q' w(q,l,q')=1
        f = open(adr, "r")
        f.readline()
        line = f.readline()
        # load dinit
        while line and line.find("F") == -1:
            line = line.replace('(', ' ').replace(')', ' ')
            l = line.split()
            q = int(l[0])
            states.add(q)
            dinit[q] = float(l[1])
            line = f.readline()
        line = f.readline()
        # load dfinal
        while line and line.find("S") == -1:
            line = line.replace('(', ' ').replace(')', ' ').replace(',', ' ')
            l = line.split()
            q = int(l[0])
            states.add(q)
            dfinal[q] = float(l[1])
            line = f.readline()
        line = f.readline()
        # load trans1
        while line and line.find("T") == -1:
            line = line.replace('(', ' ').replace(')', ' ').replace(',', ' ')
            l = line.split()
            q = int(l[0])
            states.add(q)
            x = int(l[1])
            letters.add(x)
            trans1[(q, x)] = float(l[2])
            line = f.readline()
        line = f.readline()
        # load trans2
        while line and line.find("(") != -1:
            line = line.replace('(', ' ').replace(')', ' ').replace(',', ' ')
            l = line.split()
            q1 = int(l[0])
            states.add(q1)
            x = int(l[1])
            letters.add(x)
            q2 = int(l[2])
            states.add(q2)
            trans2[(q1, x, q2)] = float(l[3])
            line = f.readline()
        f.close()
        nbl = max(list(letters)) + 1  # size of the alphabet
        nbs = max(list(states)) + 1  # number of states
        init = np.zeros(nbs)  # initial states
        for q in dinit.keys():
            init[q] = dinit[q]
        final = np.zeros(nbs)  # final states
        for q in dfinal.keys():
            final[q] = dfinal[q]
        trans = [np.zeros([nbs, nbs]) for x in range(nbl)]
        for (q1, x, q2) in trans2:
            if (q1, x) in trans1.keys():
                trans[x][q1, q2] = trans2[q1, x, q2] * trans1[q1, x] *\
                                   (1-final[q1])
        A = Automaton(nbl, nbs, init, final, trans)
        return A

    def calc_prefix_completion_weights(self, prefix):
        """ For the SPiCe competition for instance

        - Input:

        :param Automaton self: Be careful that A should be a prefix transformation of an Automata.
                               (see :func:`~automaton.Automaton.transformation`)
        :param List prefix: list of integers representing a prefix

        - Output:

        :returns: a dictionary with all alphabet letters as keys. The
                  associated values are the weights of being the next letter.
        :rtype: dict

        """
        if not self.isAbsConv:
            raise ValueError("The automaton absolute convergence is not true.")
        # Symbol -1 corresponds to the end of the sequence
        # If the weight is negative it does not carry any semantic
        p_w = self.val(prefix)
        toReturn = {}
        for i in range(self.nbL):
            w = self.val(prefix+[i])
            p_w -= w
            toReturn.update({i: max(w, 0)})
        toReturn.update({-1: max(p_w, 0)})
        return toReturn
    
    def get_dot(self, threshold = 0., nb_dec = 2, title = "Weighted Automata"):
        """ Return a string that contains the Automata into dot (graphviz) format
        
        :Example:

        >>> from splearn.datasets.base import load_data_sample
        >>> from splearn.tests.datasets.get_dataset_path import get_dataset_path
        >>> from splearn import Spectral
        >>> train_file = '3.pautomac_light.train'
        >>> data = load_data_sample(adr=get_dataset_path(train_file))
        >>> sp = Spectral()
        >>> sp.fit(X=data.data)
        >>> dotfile = "3.pautomac_light.train.dot"
        >>> dot = sp.Automaton.get_dot(threshold = 0.2, title = dotfile)
        >>> # To display the dot string one can use graphviz:
        >>> from graphviz import Source
        >>> src = Source(dot)
        >>> src.render(dotfile + '.gv', view=True) 

        - Input:

        :param  Automaton self
        :param float threshold for the value to keep. If \|weight\| < threshold, the 
        corresponding transition is not kept as an edge in the final dot string.
        :param int nb_dec is the number of decimals to keep for the weights.
        :param string title corresponds to the top comment of the string
        
        :returns: a string with the current Automata in dot format 
        """
        prec = ".{:d}f".format(nb_dec)
        out = "//{:s}\ndigraph {{\n".format(title)
        for i in range(self.nbS):
            if np.abs(self.initial[i]) > threshold and np.abs(self.final[i]) > threshold:
                label = "{0:d}\n______\n> {1:" + prec + "}\n{2:" + prec + "} >"
                label = label.format(i, self.initial[i], self.final[i])
                out += "\t{0:d} [label=\"".format(i)
                out += label + "\"]\n"
            elif np.abs(self.initial[i]) > threshold:
                label = "{0:d}\n______\n> {1:" + prec + "}"
                label = label.format(i, self.initial[i])
                out += "\t{0:d} [label=\"".format(i)
                out += label + "\"]\n"
            elif np.abs(self.final[i]) > threshold:
                label = "{0:d}\n______\n{1:" + prec + "} >"
                label = label.format(i, self.final[i])
                out += "\t{0:d} [label=\"".format(i)
                out += label + "\"]\n"
            else:                
                label = "{0:d}\n______"
                label = label.format(i)
                out += "\t{0:d} [label=\"".format(i)
                out += label + "\"]\n"
        for l in range(self.nbL):
            for i in range(self.nbS):
                for j in range(self.nbS):
                    weight = self.transitions[l][i,j]
                    if np.abs(weight) > threshold:
                        label = "{0:d}:{1:" + prec + "}"
                        label = label.format(l,weight)
                        out += "\t{0:d} -> {1:d} [label=\"".format(i,j)
                        out += label + "\"]\n"
        out += "}\n"
        return out

    @staticmethod
    def write(automaton_in, filename, format='json'):
        """ write input automaton into a file with the given format.

        - Input:

        :param Automaton automaton_in: automaton to write into the file
        :param str filename: the name of the file. If it does not exist,
         the file is created.
        :param str format: 'json' or yaml'
        """
        from splearn.serializer import Serializer
        if format == 'json':
            data_str = Serializer.data_to_json(automaton_in)
        elif format == 'yaml':
            data_str = Serializer.data_to_yaml(automaton_in)
        else:
            raise ValueError("Invalid input format. Should be \"json\" or \"yaml\"")
        with open(filename, 'w') as outfile:
            outfile.write(data_str)

    @staticmethod
    def read(filename, format='json'):
        """ return an Automaton build with attributes read from a file

        - Input:

        :param str filename: the name of the input file.
        :param str format: 'json' or yaml'

        - Output:

        :returns: the output automaton
        :rtype: Automaton
        """
        from splearn.serializer import Serializer
        with open(filename) as infile:
            datastr = infile.read()
        if format == 'json':
            return Serializer.json_to_data(datastr)
        if format == 'yaml':
            return Serializer.yaml_to_data(datastr)
        raise ValueError("Invalid input format. Should be \"json\" or \"yaml\"")
        