import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import random 
from splearn.datasets.data_sample import SplearnArray

def choose_preference(weight_0, weight_1):
    """
    Selects preference according to BT model
    INPUTS
    weight_0
    weight_1
    OUTPUTS
    switch_order - if true then prefer weight_1 over weight_0 
                    otherwise prefer weight_0 over weight_1
    """
    weight_0 = np.float64(weight_0)
    weight_1 = np.float64(weight_1)

    p = random.uniform(0,1)

    p01 = np.exp(weight_0)/( np.exp( weight_0 ) + np.exp( weight_1 ) )

    if p <= p01:
        switch_order = False
    else:
        switch_order = True 
    
    return switch_order

def construct_splearnarray(paths_list, weights_list):
    """
    INPUTS
    paths_list - [list1, list2, ..., listn] where listi = [1 5 0 8 5 4]
    weights_list - weights_list[i] = WFA(listi)
    OUTPUTS
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
    """
    if len(paths_list) != len(weights_list):
        raise ValueError("paths_list and weights_list must have the same length")

    length_list = [len(lst) for lst in paths_list]

    max_word_length = max(length_list)
    num_words = len( weights_list )

    array_words = -1*np.ones((num_words, max_word_length))
    for index, lst in enumerate( paths_list ):
        len_word   = length_list[index]
        word_array = np.array(lst)
        array_words[index,:len_word] = word_array

    splearn_array_words = SplearnArray(array_words)
    return splearn_array_words, num_words