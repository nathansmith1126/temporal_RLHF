import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import choix
import numpy as np
import random
from AUTOMATA.auto_funcs import dfa_T1, create_wfa_T1
from utils_RLHF.misc import choose_preference, construct_splearnarray
from splearn.datasets.base import DataSample
from splearn.spectral import Spectral

# symbols from automaton alphabet
symbols = ['pickup key', 'opened door', 
            'dropped key', 'closed door', 
            'pickup box', 'movement']

# parameters for WFA that encapsulates temporal task
f = 1.2
s = 0.5

# paramter for BT MLE algorithm from choix
alpha = 0.1

# wfa of desired behavior
wfa_T1 = create_wfa_T1(s=s, f=f)

num_samples = 3                   # Number of samples
max_length = 10                   # max length of each sample
num_letters = len(symbols)        # Range to permute
num_pairs   = 3

# Initially empty and will be a list of tuples in form [ (winner, loser), ...]
preference_list = []

"""
Generate random strings of natural numbers with random length
string length <= max_length
mad value of number in string == num_letters - 1 
since 0 is mapped to a letter
"""
paths_list_num = [np.random.permutation( random.randint(0, num_letters) ).tolist()[:max_length] for _ in range(num_samples)]

# Map each list of integers to a list of corresponding strings
samples_list_strings = [[symbols[i] for i in lst] for lst in paths_list_num]

# create pairs as a list of lists [[a,b], [a,c], ...]
list_pairs = [[random.randint(0, num_samples-1), random.randint(0, num_samples-1)] for _ in range(num_pairs)]

# initialize list of weights
weights_list = []
for word in samples_list_strings:
    weight = wfa_T1.weight(word=word)
    weights_list.append(weight)

# set preferences
for pair in list_pairs:
    # obtain indeces of paths in the pair
    sample_index_0 = pair[0]
    sample_index_1 = pair[1]

    # obtain weights of paths from the pair
    weight_0 = weights_list[sample_index_0]
    weight_1 = weights_list[sample_index_1]

    # determine if we should switch order
    switch_order = choose_preference(weight_0=weight_0,
                                     weight_1=weight_1 
                                     )
    if switch_order:
        # switch order
        preference_tuple = (sample_index_1, sample_index_0)
    else:
        # do not switch order
        preference_tuple = (sample_index_0, sample_index_1)
    # append preferences with tuple of properly ordered pair
    preference_list.append(preference_tuple)

# learn weights
params = choix.ilsr_pairwise(num_samples, preference_list, alpha=alpha)
# print(params)

splearn_array = construct_splearnarray(paths_list=paths_list_num, weights_list=params)

# construct data
data_tuple  = (num_samples, num_letters, splearn_array)
data_sample = DataSample(data_tuple)

# learn wfa from samples
num_hank_rows    = max_length
num_hank_columns = max_length

est = Spectral()
est.set_params(lrows=num_hank_rows, lcolumns=num_hank_columns, 
               smooth_method="trigram" ,
                version="factor" )
est.fit(data_sample)

# Print the result for debugging
for i, lst in enumerate(paths_list_num):
    print(f"List {i+1}: {lst}")
    print(f"list{i+1}: {samples_list_strings[i]}")
    print(f"weight{i+1}: {weights_list[i]}")

for i, lst in enumerate(list_pairs):
    print(f"pair {i+1}: {lst}")
    print(f"preference tuple {i+1}:{ preference_list[i] }")
    