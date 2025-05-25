import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import choix
import numpy as np
import random
from AUTOMATA.auto_funcs import *
from utils_RLHF.misc import *
# from splearn.datasets.base import DataSample
# from splearn.spectral import Spectral

if __name__ == "__main__":
    # alphabet from automaton alphabet
    # alphabet = ['pickup key', 'opened door', 
    #             'dropped key', 'closed door', 
    #             'pickup box', 'movement']
    """
    ONLY PUT REALLY GREAT PATHS INTO SPLEARN
    REPLACE MOVEMENT WITH USELESS ACTIONS 
    """

    # parameters for WFA that encapsulates temporal task
    f = 1.2
    s = 0.5
    
    # create WFA rep of 1st temporal task
    wfa_T1a = create_wfa_T1a(f=f, s=s)
    T1_alphabet = wfa_T1a.alphabet

    # structured sample set 
    max_length  = 10
    good_reps   = int( 1e3 )
    random_reps = int( 1e3 )
    num_pairs   = int( 1e4 )
    train_words_num = T1_samples_num(good_reps=good_reps, 
                                     random_reps=random_reps, 
                                     max_length=max_length)

    # create WFA for simple debug
    simple_wfa = simple_wfa()
    simple_alphabet = simple_wfa.alphabet

    # paramter for BT MLE algorithm from choix
    alpha = 0.1

    # parameters for spectral learning 
    num_hank_rows    = int(10)
    num_hank_columns = int(10)
    rank = 6
    show_WFA = True
    scale_factor     = 10

    BT_SPEC_EST = BT_SPEC_Estimator(WFA=wfa_T1a, rank=rank, 
                                    max_length=max_length, 
                                    num_pairs=num_pairs,
                                    num_hank_rows=num_hank_rows, 
                                    num_hank_columns=num_hank_columns,
                                    scale_factor=scale_factor,
                                    alpha=alpha,
                                    train_words_num=train_words_num,
                                    show_WFA=show_WFA)
    BT_SPEC_EST.create_pairs()
    BT_SPEC_EST.pairs2weights()
    BT_SPEC_EST.construct_splearnarray()
    BT_SPEC_EST.build_spec_est()

    for index, word in enumerate( BT_SPEC_EST.words_list_string ):
        lweight     = BT_SPEC_EST.learned_WFA.weight(word)
        true_weight = wfa_T1a.weight(word)
        BT_score    = BT_SPEC_EST.weights_BT[index]

    # est, sup, _, data_splearn_unique, params_BT, weights_list, words_list_string = test_pref_spec_learn(WFA=wfa_T1a,
    #                                                                                         rank=rank,
    #                                                                                         max_length=max_length, 
    #                                                                                         num_pairs=num_pairs, 
    #                                                                                         alpha=alpha, show_WFA=show_WFA, 
    #                                                                                         num_hank_columns=num_hank_columns, 
    #                                                                                         num_hank_rows=num_hank_rows, 
    #                                                                                         scale_factor=scale_factor, 
    #                                                                                         train_words_num=train_words_num)

    # est_weights   = est.predict(data_splearn_unique)
    # train_weights = np.array(weights_list)
    # # diff = (est_weights - train_weights).astype(float)
    # # error = np.sum(np.abs( diff ) )/len(weights_list)
    # # print(f"Normalized error is {error}")

    # splearn_WFA = est.automaton
    # WFA = spwfa2WFA(splearn_WFA, alphabet=T1_alphabet)

    # sup_splearn_WFA = sup.automaton
    # sup_WFA = spwfa2WFA(sup_splearn_WFA, alphabet=T1_alphabet)

    # t0 = splearn_WFA.initial
    # transitions = splearn_WFA.transitions
    # tf = splearn_WFA.final 

    # for index, word in enumerate( words_list_string ):
    #     lweight     = WFA.weight(word)
    #     lsup_weight = sup_WFA.weight(word)
    #     true_weight = wfa_T1a.weight(word)
    #     BT_score    = params_BT[index]

"""
SUPPLEMENTARY CODE
""" 

# def test_pref_spec_learn(WFA,
#                          rank, 
#                          max_length, num_pairs, 
#                          alpha, 
#                          num_hank_rows, 
#                          num_hank_columns, 
#                          scale_factor, 
#                          train_words_num = None, 
#                          show_WFA = False):
#     """
#     Test RLHF pipeline that takes as input WFA and samples preferences from WFA
#     to learn an approximation WFA_hat.

#     INPUTS
#     WFA (WeightedAutomaton): - instance of weighted_automaton class
#     max_length - max length of sample word
#     num_pairs - number of ordered pairs randomly generated to be assigned weights in 
#                 accordance with BT model as weight(word) = score for BT model
#     alpha - hyperparameter for choix BT MLE algorithm
#     num_hank_rows - the m in m x n hankel matrix created by spectral.fit method
#     num_hank_columns - the n in m x n hankel matrix created by spectral.fit method
#     show_WFA - prints WFA learned from data if true, set to false on default
#     OUTPUTS
#     est - instance of estimator class with WFA attached from scikit-splearn
#     inp - data input for est.fit() method that is unscaled 
#             DO NOT UNDERSTAND STRUCTURE
#             it housees num_samples, num_letters and the strings themselves
#     inp_scaled - data input for est.fit() method 
#             that is scaled by the weights calculated with BT inferencing 
#             DO NOT UNDERSTAND STRUCTURE
#             it housees num_samples, num_letters and the strings themselves
#     params_BT - parameters learned by BT 
#     weights_list - weights used to make preferences
#     """

#     alphabet = WFA.alphabet

#     num_letters = len( alphabet )

#     # Initially empty and will be a list of tuples in form [ (winner, loser), ...]
#     preference_list = []

#     # list of words derived from random pairs in numeric form and string form
#     words_list_num = []
#     words_list_string = []
#     # weights of words derived from random pairs that s
#     weights_list = []

#     # Map each list of integers to a list of corresponding strings
#     # words_list_strings = [[alphabet[i] for i in lst] for lst in words_list_num]

#     # create pairs as a list of lists [[a,b], [a,c], ...]
#     # list_pairs = [[random.randint(0, num_samples-1), random.randint(0, num_samples-1)] for _ in range(num_pairs)]

#     # for pair in np.arange(num_pairs):
#     #     word_num_0 = np.random.permutation( random.randint(0, num_letters) ).tolist()[:max_length]
#     #     word_num_1 = np.random.permutation( random.randint(0, num_letters) ).tolist()[:max_length]
    
#     for _ in range( num_pairs ):
#         """
#         Generate random strings of natural numbers with random length
#         string length <= max_length
#         mad value of number in string == num_letters - 1 
#         since 0 is mapped to a letter
#         """
#         if train_words_num is None:
#             '''
#             randomly create a word to be trained on since no list is provided 
#             '''
#             word_num_0 = [np.random.randint(0, num_letters) for _ in np.arange( np.random.randint(0, max_length) )] 
#             word_num_1 = [np.random.randint(0, num_letters) for _ in np.arange( np.random.randint(0, max_length) )] 
#         else:
#             '''
#             Randomly select words for pair from input list
#             '''
#             word_num_0 = random.choice(train_words_num)
#             word_num_1 = random.choice(train_words_num)
#         # string representation of words
#         word_string_0 = [alphabet[index_letter] for index_letter in word_num_0]
#         word_string_1 = [alphabet[index_letter] for index_letter in word_num_1]

#         # obtain weights of paths from the pair
#         weight_0 = WFA.weight(word_string_0)
#         weight_1 = WFA.weight(word_string_1)

#         # add words to training list if it has not yet been seen
#         if word_num_0 not in words_list_num:
#             # add word_0 to list since it has not been included yet
#             words_list_num.append(word_num_0)
#             weights_list.append(weight_0)
#             words_list_string.append(word_string_0)
#         if word_num_1 not in words_list_num:
#             # add word_1 to list since it has not been included yet
#             words_list_num.append(word_num_1)
#             weights_list.append(weight_1)
#             words_list_string.append(word_string_1)
#         """
#         Need samples in specific form. 
#         Need a bijective labeling on selected words 
#         where each orderrf pair contains indices from this labeling
#         """

#         # new indices needed to properly organize data for choix BT MLE alg
#         new_index_0 = words_list_num.index(word_num_0)
#         new_index_1 = words_list_num.index(word_num_1)

#         # determine if we should switch order
#         switch_order = choose_preference(
#                                         weight_0=weight_0,
#                                         weight_1=weight_1 
#                                         )
#         if switch_order:
#             # switch order
#             preference_tuple = (new_index_1, new_index_0)
#         else:
#             # do not switch order
#             preference_tuple = (new_index_0, new_index_1)
#         # append preferences with tuple of properly ordered pair
#         preference_list.append(preference_tuple)   

#     # unique words obtained from analyzing random pairs
#     num_words_unique = len(words_list_num)
    
#     # learn weights
#     params_BT = choix.ilsr_pairwise(num_words_unique, preference_list, alpha=alpha)
#     # print(params_BT)

#     # scale weights to be > 0
#     updated_params_BT = params_BT + np.abs( np.min( params_BT ) )
#     splearn_array_unique, _, splearn_array_scaled, num_words_scaled = construct_splearnarray(words_list=words_list_num, 
#                                                                                              weights_list=updated_params_BT, 
#                                                                                              scale_factor=scale_factor)

#     # construct data for training and unique data for score evaluation
#     data_tuple_scaled  = (num_letters, num_words_scaled, splearn_array_scaled)
#     data_tuple_unique  = (num_letters, num_words_unique, splearn_array_unique)

#     # map tuples to data sample form for scikit splearn package
#     data_sample_scaled = DataSample(data_tuple_scaled)
#     data_sample_unique = DataSample(data_tuple_unique)

#     # obtain data object from data samples 
#     inp_scaled = data_sample_scaled.data
#     inp_unique = data_sample_unique.data 

#     est = Spectral()
#     est.set_params(lrows=num_hank_rows, lcolumns=num_hank_columns, 
#                 smooth_method="trigram" , rank=rank,
#                     version="factor" )
    
#     est_sup = Spectral()
#     est_sup.set_params(lrows=num_hank_rows, lcolumns=num_hank_columns, 
#                 smooth_method="trigram" , rank=rank,
#                     version="factor" )
    
#     est.get_params()
#     # Spectral(full_svd_calculation=False , 
#     #             lcolumns=num_hank_columns, lrows =num_hank_rows,
#     #             mode_quiet=False , partial=True , rank =5,
#     #             smooth_method="trigram" , sparse=True ,
#     #             version= "factor" )
#     est.fit(inp_scaled)
#     est_sup.fit(inp_unique, y=updated_params_BT)
    
#     if show_WFA:
#         print(f"initial: {est.automaton.initial}")
#         print(f"final: {est.automaton.final}")
#         print(f"transitions: {est.automaton.transitions}")
    
#     return est, est_sup, inp_scaled, inp_unique, params_BT, weights_list, words_list_string
