The **scikit-splearn** package is a  Python scikit toolbox for spectral learning algorithms.

These algorithms aim at learning Weighted Automata (WA) using what is named a Hankel matrix. The toolbox thus provides also a class for WA (with a bunch of useful methods), another one for Hankel matrix, and a class for loading data. As WA are a generalization of classical Probabilistic Automaton (and thus HMM), everything works for these simpler models.

The core of the learning algorithms is to compute a singular values decomposition of the Hankel matrix and then to construct the weighted automata from the elements of the decomposition. This is done in the class Learning.

In its classic version, the rows of the Hankel matrix are prefixes while its columns are suffixes. Each cell contains then the probability of the sequence starting with the corresponding prefix and ending with the corresponding suffix. In the case of learning, the cells contain observed frequencies. **scikit-splearn** provides other versions, where each cell contains the probability that the corresponding sequence is prefix, a suffix, or a factor.

Formally, the Hankel matrix is bi-infinite. Hence, in case of learning, one has to concentrate on a finite portion. The parameters lrows and lcolumn allows to specified which subsequences are taken into account as rows and columns of the finite matrix. If, instead of a list, an integer is provided, the finite matrix will have all rows and columns that correspond to subsequences up to these given lengths. 

The learning method requires also the information about the rank of the matrix. This rank corresponds to the number of states of a minimal WA computing the matrix (in case of learning, this is the estimated number of states of the target automaton). There is no easy way to evaluate the rank, a cross-validation approach is usually used to find the best possible value.

Finally, **scikit-splearn** provides 2 ways to store the Hankel matrix: a classical one as an array, and a sparse version using ``scipy.sparse``.

The original scikit-splearn Toolbox is developed in Python at `LabEx Archimède <http://labex-archimede.univ-amu.fr/>`_ , as a `LIS <http://www.lis-lab.fr/>`_ project.

This package, as well as the **scikit-splearn** toolbox, is Free software, released under BSD License.

The latest version of **scikit-splearn** can be downloaded from the following
`PyPI page <https://pypi.python.org/pypi/scikit-splearn/>`_ .

The documentation is available `here <http://dev.pages.lis-lab.fr/scikit-splearn>`_ .

There is also a `gitlab repository <https://gitlab.lis-lab.fr/dev/scikit-splearn.git>`_ , which provides the git repository managing the source code and where issues can be reported.
