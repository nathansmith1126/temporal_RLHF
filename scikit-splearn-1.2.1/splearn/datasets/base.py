import pickle
import numpy as np
from splearn.datasets.data_sample import DataSample


def load_data_sample(adr, filetype='SPiCe', pickle=False):
    """Load a sample from file and returns a dictionary
    (word,count)

    - Input:

    :param str adr: address and name of the loaded file
    :param str filetype: (default value = 'SPiCe') indicate
           the structure of the file. Should be either 'SPiCe' or 'Pautomac'
    :param boolean pickle: if enabled it a pickle file is created from the loaded file. Default is fault.

    - Output:

    :returns: corresponding DataSample
    :rtype: DataSample


    :Example:

    >>> from splearn.datasets.base import load_data_sample
    >>> from splearn.tests.datasets.get_dataset_path import get_dataset_path
    >>> train_file = '3.pautomac_light.train' # '4.spice.train'
    >>> data = load_data_sample(adr=get_dataset_path(train_file))
    >>> data.nbL
    4
    >>> data.nbEx
    5000
    >>> data.data
    Splearn_array([[ 3.,  0.,  3., ..., -1., -1., -1.],
           [ 3.,  3., -1., ..., -1., -1., -1.],
           [ 3.,  2.,  0., ..., -1., -1., -1.],
           ...,
           [ 3.,  1.,  3., ..., -1., -1., -1.],
           [ 3.,  0.,  3., ..., -1., -1., -1.],
           [ 3.,  3.,  1., ..., -1., -1., -1.]])

    """

    if filetype == 'SPiCe' or filetype == 'Pautomac':
        data = _load_file_doublelecture(adr=adr, pickle=pickle)
        return DataSample(data=data)

def _load_file_doublelecture(adr, pickle=False):
    dsample = {}  # dictionary (word,count)
    _, max_length = _read_dimension(adr=adr)
    f = open(adr, "r")
    line = f.readline()
    l = line.split()
    nbEx = int(l[0])
    nbL = int(l[1])
    line = f.readline()
    data1 = np.zeros((nbEx, max_length ))
    data1 += -1
    i = 0
    while line:
        l = line.split()
        # w = () if int(l[0]) == 0 else tuple([int(x) for x in l[1:]])
        # dsample[w] = dsample[w] + 1 if w in dsample else 1
        # traitement du mot vide pour les préfixes, suffixes et facteurs
        w = [] if int(l[0]) == 0 else [int(x) for x in l[1:]]
        data1[i, :len(w)] = w
        line = f.readline()
        i += 1
    # print("data1 ", data1)
    f.close()
    if pickle:
        _create_pickle_files(adr=adr, dsample=dsample)
    return nbL, nbEx, data1

def _read_dimension(adr):
    f = open(adr, "r")
    line = f.readline()
    l = line.split()
    nbEx = int(l[0])
    nbL = int(l[1])
    line = f.readline()
    max_length = 0
    nb_sample = 0
    while line:
        l = line.split()
        nb_sample += 1
        length = int(l[0])
        if max_length < length:
            max_length = length
        line = f.readline()
    f.close()
    if nb_sample != nbEx:
        raise ValueError("check imput file, metadata " + str(nbEx) +
                         "do not match number of samples " + str(nb_sample))
    return nb_sample , max_length

def _create_pickle_files(self, adr, dsample):
    f = open(adr + ".sample.pkl", "wb")
    pickle.dump(dsample, f)
    f.close()
