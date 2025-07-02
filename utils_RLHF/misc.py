import sys
from datetime import datetime
import os
import pickle 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import choix 
import numpy as np
import random 
import sympy as sp
import gymnasium as gym 
from gymnasium.envs.registration import register
from splearn.datasets.data_sample import SplearnArray
from splearn.datasets.base import DataSample
from splearn.spectral import Spectral
from Minigrid.minigrid.envs.test_envs import ordered_obj, WFA_TestEnv 
from AUTOMATA.auto_funcs import spwfa2WFA, create_wfa_T1a, WFA_monitor, dfa_T1
from WA_package.weighted_automaton import WeightedAutomaton
from typing import Optional 

class BT_SPEC_Estimator:
    def __init__(self):
        """
        Test RLHF pipeline that takes as input WFA and samples preferences from WFA
        to learn an approximation WFA_hat.

        INPUTS
        WFA (WeightedAutomaton): - instance of weighted_automaton class
        max_length - max length of sample word
        num_pairs - number of ordered pairs randomly generated to be assigned weights in 
                    accordance with BT model as weight(word) = score for BT model
        alpha - hyperparameter for choix BT MLE algorithm
        num_hank_rows - the m in m x n hankel matrix created by spectral.fit method
        num_hank_columns - the n in m x n hankel matrix created by spectral.fit method
        show_WFA - prints WFA learned from data if true, set to false on default
        USEFUL ATTRIBUTES
        est - instance of estimator class with WFA attached from scikit-splearn
        inp - data input for est.fit() method that is unscaled 
                DO NOT UNDERSTAND STRUCTURE
                it housees num_samples, num_letters and the strings themselves
        inp_scaled - data input for est.fit() method 
                that is scaled by the weights calculated with BT inferencing 
                DO NOT UNDERSTAND STRUCTURE
                it housees num_samples, num_letters and the strings themselves
        weights_BT - weightss learned by BT from preferences
        weights_list - weights used to make preferences
        """
        self.WFA  = None
        self.rank = None
        self.max_length = None
        self.num_pairs  = None
        self.num_hank_rows = None
        self.num_hank_columns = None
        self.scale_factor = None
        self.alpha = None
        self.train_words_num = None
        self.show_WFA = None
        self.num_letters = None

        # Initially empty and will be a list of tuples in form [ (winner, loser), ...]
        self.preference_list = []

        # list of words derived from random pairs in numeric form and string form
        self.words_list_num = []
        self.words_list_string = []
        
        # weights of words derived from random pairs that s
        self.weights_list = []

        # weights learned from preferences by BT
        self.weights_BT = []

        self.num_words_scaled = None

        # special arrays used for input into splearn estimator
        self.splearn_array_unique_words = None
        self.splearn_array_scaled = None 

        # estimator from scaled data
        self.spec_EST = None 

        # Learned WFA from speclearn in sympy form of WeightedAutomaton 

    # (self, WFA: WeightedAutomaton,
    #                     max_length: int , num_pairs: int,  
    #                     rank: int, num_hank_rows: int, 
    #                     num_hank_columns: int, scale_factor: int, 
    #                     Method: str, alpha: Optional[float] = None, 
    #                     train_words_num: Optional[int]=None, 
    #                     show_WFA: Optional[bool] = False)
    def create_pairs(self, WFA: WeightedAutomaton,
                        max_length: int , num_pairs: int,  
                        train_words_num: Optional[int]=None 
                        ):
        """
        Generate random strings of natural numbers with random length
        string length <= max_length
        mad value of number in string == num_letters - 1 
        since 0 is mapped to a letter
        """
        self.WFA  = WFA
        self.max_length = max_length
        self.num_pairs  = num_pairs
        self.train_words_num = train_words_num
        self.num_letters = len( self.WFA.alphabet )
        for _ in range( self.num_pairs ):
            if self.train_words_num is None:
                '''
                randomly create a word to be trained on since no list is provided 
                '''
                word_num_0 = [np.random.randint(0, self.num_letters) for _ in np.arange( np.random.randint(0, self.max_length) )] 
                word_num_1 = [np.random.randint(0, self.num_letters) for _ in np.arange( np.random.randint(0, self.max_length) )] 
            else:
                '''
                Randomly select words for pair from input list
                '''
                word_num_0 = random.choice(self.train_words_num)
                word_num_1 = random.choice(self.train_words_num)
            # string representation of words
            word_string_0 = [self.WFA.alphabet[index_letter] for index_letter in word_num_0]
            word_string_1 = [self.WFA.alphabet[index_letter] for index_letter in word_num_1]

            # obtain weights of paths from the pair
            weight_0 = self.WFA.weight(word_string_0)
            weight_1 = self.WFA.weight(word_string_1)

            # add words to training list if it has not yet been seen
            if word_num_0 not in self.words_list_num:
                # add word_0 to list since it has not been included yet
                self.words_list_num.append(word_num_0)
                self.weights_list.append(weight_0)
                self.words_list_string.append(word_string_0)
            if word_num_1 not in self.words_list_num:
                # add word_1 to list since it has not been included yet
                self.words_list_num.append(word_num_1)
                self.weights_list.append(weight_1)
                self.words_list_string.append(word_string_1)
            """
            Need samples in specific form. 
            Need a bijective labeling on selected words 
            where each orderrf pair contains indices from this labeling
            """

            # new indices needed to properly organize data for choix BT MLE alg
            new_index_0 = self.words_list_num.index(word_num_0)
            new_index_1 = self.words_list_num.index(word_num_1)

            # determine if we should switch order
            switch_order = self.choose_preference(
                                            weight_0=weight_0,
                                            weight_1=weight_1 
                                            )
            if switch_order:
                # switch order
                preference_tuple = (new_index_1, new_index_0)
            else:
                # do not switch order
                preference_tuple = (new_index_0, new_index_1)
            # append preferences with tuple of properly ordered pair
            self.preference_list.append(preference_tuple)   

        # unique words obtained from analyzing random pairs
        self.num_words_unique = len(self.words_list_num)

    def choose_preference(self, weight_0, weight_1):
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

    def pairs2weights(self, alpha: Optional[float] = 0.1):
        """
        Returns weights that maximize likelihood
        Arg:
        alpha [float] - regularization parameter
        """
        self.alpha = alpha 
        # learn weights
        weights_BT = choix.ilsr_pairwise(self.num_words_unique, 
                                        self.preference_list, 
                                        alpha=self.alpha)
        # print(weights_BT)

        # scale weights to be > 0 (does not affect probabilities)
        self.weights_BT = weights_BT + np.abs( np.min( weights_BT ) )

    def construct_splearnarray(self, scale_factor: Optional[int] = 10, Method: Optional[str] = "proportional",
                                top_scale: Optional[int] = None):
        """
        INPUTS
        words_list - [list1, list2, ..., listn] where listi = [1 5 0 8 5 4] each list should be unique
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
        self.scale_factor = scale_factor 
        self.Method = Method

        if len(self.words_list_num) != len(self.weights_BT):
            raise ValueError("words_list and weights_BT must have the same length")

        length_list = [len(word) for word in self.words_list_num]

        max_word_length = max(length_list)
        num_words = len( self.weights_BT )

        # array housing words in splearn format without repeat
        # will be filled with for loop
        array_words = -1*np.ones((num_words, max_word_length))
        if Method =="top_score_only":
            if top_scale == None:
                top_scale = 10*self.num_words_unique
            # list of ordered pairs [ (prev_index, top_score), (prev_idex, 2nd_best_score), ...,(prev_index, worst_score)]
            indexed_weights = scores2samples(self.weights_BT)
            best_score      = indexed_weights[0]
            word_index      = best_score[0]
            word       = self.words_list_num[word_index]
            len_word   = length_list[word_index]
            word_array = np.array(word)
            array_words[0,:len_word] = word_array 
            word_extended = array_words[0,:]
            # top score comes from word_extended and scaled array has yet to be created
            array_words_scaled = self.repeat_construct(prefix_word_array=word_extended, 
                                                               len_prefix=len_word)
        
        elif Method == "top_score":
            if top_scale == None:
                top_scale = 10*self.num_words_unique
            # list of ordered pairs [ (prev_index, top_score), (prev_idex, 2nd_best_score), ...,(prev_index, worst_score)]
            indexed_weights = scores2samples(self.weights_BT)
            for ind, indexed_weight in enumerate( indexed_weights ):
                word_index = indexed_weight[0]
                len_word   = length_list[word_index]
                word       = self.words_list_num[word_index]
                word_array = np.array(word)
                array_words[ind,:len_word] = word_array 
                word_extended = array_words[ind,:]

                if ind == 0:
                    # top score comes from word_extended and scaled array has yet to be created
                    array_words_scaled = self.repeat_construct(prefix_word_array=word_extended, 
                                                               len_prefix=len_word)
                else:
                    array_words_scaled = np.vstack((array_words_scaled, word_extended))

        elif Method == "proportional":
            for index, word in enumerate( self.words_list_num ):
                len_word   = length_list[index]
                word_array = np.array(word)
                array_words[index,:len_word] = word_array 
                weight_word = self.weights_BT[index]

                # extended word filled with -1 to be put into scaled array 
                # where the number of repeats is dependent on the BT score
                word_extended = array_words[index,:]

                if weight_word >= 1/scale_factor:
                    # big enough to qualify for repeats
                    extra_reps = int( np.round( (scale_factor)*weight_word ) )
                    scaled_array = np.tile(word_extended, (extra_reps, 1) )
                    if index == 0:
                        array_words_scaled = scaled_array
                        first_qualifier = False
                    else:
                        array_words_scaled = np.vstack((array_words_scaled, scaled_array))
                else:
                    # not big enough to qualify for repeats
                    # scaled_array = word_extended
                    pass 
                # if first_qualifier:
                #     array_words_scaled = scaled_array
                #     first_qualifier = False
                # else:
                #     array_words_scaled = np.vstack((array_words_scaled, scaled_array))

        self.num_words_scaled, _ = array_words_scaled.shape
        self.splearn_array_unique = SplearnArray(array_words)
        self.splearn_array_scaled = SplearnArray(array_words_scaled)

    def repeat_construct(self,prefix_word_array: list[int], 
                         len_prefix: int, scale: Optional[int] = None):
        
        if scale == None:
            scale = 10*self.num_words_unique

        # prefix_array = np.tile(top_word_extended, (len_best, 1) )

        for ind in range(len_prefix,0,-1):

            # create prefixes
            prefix_word_array[ind:] = - 1
            prefix_array_scaled = np.tile(prefix_word_array, (scale*ind, 1) ) 

            if ind == len_prefix:
                prefix_array_complete = prefix_array_scaled
            else:
                prefix_array_complete = np.vstack( 
                                                (prefix_array_complete, 
                                                prefix_array_scaled) 
                                                )
        
        
        return prefix_array_complete

    def build_spec_est(self, rank: int, num_hank_rows: int, 
                        num_hank_columns: int, show_WFA: Optional[bool] = False):
        
        self.rank = rank 
        self.num_hank_rows = num_hank_rows
        self.num_hank_columns = num_hank_columns
        self.show_WFA = show_WFA
        # unique labels not used at this time because splearn sucks testacles
        # construct data for training and unique data for score evaluation
        data_tuple_scaled  = (self.num_letters, self.num_words_scaled, self.splearn_array_scaled)
        # data_tuple_unique  = (self.num_letters, self.num_words_unique, self.splearn_array_unique)

        # map tuples to data sample form for scikit splearn package
        data_sample_scaled = DataSample(data_tuple_scaled)
        # data_sample_unique = DataSample(data_tuple_unique)

        # obtain data object from data samples 
        inp_scaled = data_sample_scaled.data
        # inp_unique = data_sample_unique.data 

        # initialize spectral estimator
        est = Spectral()
        est.set_params(lrows=self.num_hank_rows, lcolumns=self.num_hank_columns, 
                    smooth_method="trigram" , rank=self.rank,
                        version="factor" )
        
        # apply spectral learning to scaled data 
        est.fit(inp_scaled)
        # est_sup.fit(inp_unique, y=updated_params_BT)
        
        self.spec_EST = est
        self.hankels = self.spec_EST._hankel
        splearn_WFA = est.automaton
        self.learned_WFA = spwfa2WFA(splearn_WFA, alphabet=self.WFA.alphabet)

        if self.show_WFA:
            print(f"initial: {est.automaton.initial}")
            print(f"final: {est.automaton.final}")
            print(f"transitions: {est.automaton.transitions}")
    
def T1_samples_num(good_reps, random_reps, max_length):

    num_letters = 6
    train_words_num = [ [np.random.randint(0, num_letters) for _ in np.arange( np.random.randint(0, max_length) )] for _ in np.arange(random_reps) ]

    # add good words
    l1 = [0, 1 , 2 ,3 , 4]
    l2 = [0, 5, 1, 5, 2, 5, 3, 5, 4]
    l3 = [0,5,5,1,2,3,4]
    l4 = [0, 5, 5, 1, 5, 2, 5, 5, 3, 5,5, 4]

    for _ in range(good_reps):
        train_words_num.append(l1)
        train_words_num.append(l2)
        train_words_num.append(l3)
        train_words_num.append(l4)
    
    return train_words_num

def scores2samples(scores):
    """
    INPUTS
    scores - list of scores from BT inference
    
    OUTPUTS
    scaled - 
    """
    count = len(scores)
    indexed_scores = list(enumerate(scores))
    indexed_scores.sort(key=lambda x: x[1], reverse=True) 

    return indexed_scores

def word2WFA_max(word: list[str], 
                 alphabet: list[str], 
                 f: Optional[float] = 1.2, 
                 s: Optional[float] = 0.8, 
                 u: Optional[float] = 0.75,
                 benign_events: Optional[ list[str] ] = []) -> WeightedAutomaton:
    """
    Maps a word (sigma_max) to a WFA, A where f_A(word) > f_(all other words)

    Args:
        word (list of str): list of strings corresponding to the desired word e.g. word = ["a", "b", "c"] with word[0] = "a" is the first event in the word
        alphabet (list of str): list of strings corresponding to every letter or event in big Sigma (alphabet) MUST BE NO REPEATS
        f (float): parameter to promote forward progress through word
        s (float): parameter to decrease f_A if forward progress is not made
        u (float): parameter for useless events
        benign_events: events that are irrelevant if recorded and make no change to f_A output. All the trans matrices are identity
    Returns:
        WFA_max (WeightedAutomaton): a Weighted automaton object where input word maximizes it's scoring function 
    """

    # num of states in WFA
    num_states  = len(word) + 1

    # events that are neither benign or used for progress in word are useless
    # useless_events_set = set(alphabet) - set(benign_events) - set(word)
    # useless_events = list( useless_events_set )

    # intial and final arrays of WFA
    initial_array = sp.Matrix.zeros(1,num_states)
    initial_array[0,0] = 1
    final_array   = sp.Matrix.ones(num_states,1)

    # empty transition dictionary to house matrices for each event
    transition_dictionary = {}
    
    for event_index, event in enumerate(alphabet):
        if event not in word:
            # event is either useless or benign
            if event in benign_events:
                # event is benign so it's transition matrix is identity
                transition_dictionary[event] = sp.Matrix.eye(num_states)
            else:
                # event is useless so transition matrix is eye scaled down
                transition_dictionary[event] = u*sp.Matrix.eye(num_states)
        else: 
            # event must be in word
            # find indices in word where word[indices] = event
            word_indices = [word_index for word_index in range( len(word) ) if word[word_index] == event]

            # initialize sub-matrices usd to build transition matrix for each event 
            progress_matrix   = sp.Matrix.zeros(num_states, num_states)
            adjustment_matrix = sp.Matrix.zeros(num_states, num_states)

            # now we build the transition matrix
            for word_index in word_indices:
                # fill progress_matrix with f in the right places
                progress_matrix[word_index, word_index+1] = f

                # fill adjustment matrix with stationary paremeter in the right places
                adjustment_matrix[word_index, word_index] = s

            transition_matrix = s*sp.eye(num_states) + progress_matrix - adjustment_matrix
            transition_dictionary[event] = transition_matrix

    WFA_max = WeightedAutomaton(n=num_states,alphabet=alphabet, 
                                initial=initial_array, 
                                transitions=transition_dictionary, 
                                final=final_array)
    return WFA_max

def save_bts_est(bts_est, folder_path):
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Create a timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"bts_est_{timestamp}.pkl"
    full_path = os.path.join(folder_path, filename)

    # Save the object
    with open(full_path, "wb") as f:
        pickle.dump(bts_est, f)

    print(f"Saved BTS Estimator to: {full_path}")
    return full_path  # Return the path so it can be logged or reused

def load_bts_est(pickle_path):
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"File not found: {pickle_path}")
    with open(pickle_path, "rb") as f:
        bts_est = pickle.load(f)
    print(f"Loaded BTS Estimator from: {pickle_path}")
    return bts_est

def create_BTS_EST():
    import time
    time.sleep(5)
    # parameters for WFA that encapsulates temporal task
    f = 1.2
    s = 0.5
    
    # create WFA rep of 1st temporal task
    wfa_T1a = create_wfa_T1a(f=f, s=s)
    # T1_alphabet = wfa_T1a.alphabet

    # structured sample set 
    max_length  = 10
    good_reps   = int( 1e3 )
    random_reps = int( 1e3 )
    num_pairs   = int( 1e4 )
    train_words_num = T1_samples_num(good_reps=good_reps, 
                                     random_reps=random_reps, 
                                     max_length=max_length)

    # create WFA for simple debug
    # simple_wfa = simple_wfa()
    # simple_alphabet = simple_wfa.alphabet

    # paramter for BT MLE algorithm from choix
    alpha = 0.1

    # parameters for spectral learning 
    num_hank_rows    = int(10)
    num_hank_columns = int(10)
    rank = 6
    show_WFA = True
    scale_factor     = 100

    Method = "top_score_only"
    """
    inputs for create_pairs - wfa, max_length, num_pairs, train_words (4)
    inputs for pairs2weights - alpha (1)
    inputs for construct_splearn_array - scale_factor, Method (2)
    inputs for build_spec_est - hank_rows, hank_cols, rank, show_WFA (4)
    """
   
    BT_SPEC_EST = BT_SPEC_Estimator()
    BT_SPEC_EST.create_pairs(WFA=wfa_T1a, max_length=max_length,
                             num_pairs=num_pairs,train_words_num=train_words_num)
    BT_SPEC_EST.pairs2weights(alpha=alpha)
    BT_SPEC_EST.construct_splearnarray(scale_factor=scale_factor,Method=Method)
    BT_SPEC_EST.build_spec_est(rank=rank, num_hank_rows=num_hank_rows, 
                               num_hank_columns=num_hank_columns,show_WFA=show_WFA)
    return BT_SPEC_EST

def register_special_envs( ENV, max_steps: Optional[int] = None ):

    #    wfa_monitor: Optional[WFA_monitor]=None, 
    #   f_reward: Optional[float] = None, 
    #   f_penalty: Optional[float] = None, 
    #   env_size: Optional[float]  = None, 
    #   finish_factor: Optional[float] = None,
    #   max_steps: Optional[int] = None

    ENV_NAME = ENV.registered_name
    if ENV_NAME in gym.envs.registry:
        print(f"{ENV_NAME} is already registered, no need to register")
    else:
        if ENV_NAME == "MiniGrid-TemporalSPWFATestEnv-v0":
            "register WFA augmented env"
            # f = 0.99
            # s = 0.4
            # WFA_T1 = create_wfa_T1(f=f, s=s)
            est_name = "bts_est_2025-05-22_10-30-34.pkl"
            est_direc = r"C:\Users\nsmith3\Documents\GitHub\temporal_RLHF\BTS_models"
            full_path = os.path.join(est_direc, est_name)
            BTS_EST = load_bts_est(pickle_path=full_path)
            
            WFA = BTS_EST.learned_WFA

            # register environment
            register(
                id=ENV_NAME,               # Unique environment ID
                entry_point="Minigrid.minigrid.envs.test_envs:SPWFA_TestEnv",  # Module path to the class
                kwargs={
                    "WFA": WFA,
                    "max_steps": max_steps,
                    "render_mode": "rgb_array"
                },
            )
        elif ENV_NAME == "MiniGrid-Temporal-ord_obj-v0":
                register(
                id=ENV_NAME,  # Unique environment ID
                entry_point="Minigrid.minigrid.envs.test_envs:ordered_obj",  # Module path to the class
                kwargs={
                    "wfa_monitor": ENV.wfa_monitor,
                    "objects_list": ENV.objects_list, 
                    "actions_list": ENV.actions_list, 
                    "max_steps": max_steps, 
                    "f_reward": ENV.f_reward, 
                    "f_penalty": ENV.f_penalty, 
                    "finish_factor": ENV.finish_factor,
                    "size": ENV.size, 
                    "render_mode": "rgb_array"
                },
            )
        elif ENV_NAME == "MiniGrid-TemporalTestEnv-v0":
            # register dfa environment
            register(
                id=ENV_NAME,               # Unique environment ID
                entry_point="Minigrid.minigrid.envs.test_envs:TestEnv",  # Module path to the class
                kwargs={
                    "auto_task": dfa_T1,
                    "auto_reward": 0.1,
                    "render_mode": "rgb_array"
                },
            )
        elif ENV_NAME == "MiniGrid-TemporalWFATestEnv-v0" :
            register(
                id=ENV_NAME,               # Unique environment ID
                entry_point="Minigrid.minigrid.envs.test_envs:WFA_TestEnv",  # Module path to the class
                kwargs={
                    "WFA_monitor": ENV.WFA_monitor,
                    "f_reward": ENV.f_reward,
                    "f_penalty": ENV.f_penalty,
                    "finish_factor": ENV.finish_factor,
                    "max_steps": ENV.max_steps,
                    "render_mode": "rgb_array"
                },
            )
        else:
            raise ValueError(f"Unknown environment name passed: {ENV_NAME}")

def create_ord_obj_env(word:Optional[ list[str] ] = ["pickup ball", "dropped ball", "pickup box", "dropped box"], 
                       actions_list:Optional[ list[str] ] = ["pickup", "dropped"], 
                        objects_list:Optional[ list[str] ] = ["ball", "box"], 
                         benign_events:Optional[ list[str] ] = ["pickup key", "dropped key"], 
                          f_reward: Optional[float] = 10.0,
                           f_penalty: Optional[float] = 0.25,
                            finish_factor: Optional[float] = 10.0,
                             env_size: Optional[int] = 6, 
                              f: Optional[float] = 1.2, 
                               s: Optional[float] = 0.8, 
                                u: Optional[float] = 0.75,
                                render_mode: Optional[ str ] = "rgb_array"
                              ) -> ordered_obj:
    """
    Creates ordered_object environment

    Args:
        word (list of str): list of strings corresponding to the desired word e.g. word = ["a", "b", "c"] with word[0] = "a" is the first event in the word
        alphabet (list of str): list of strings corresponding to every letter or event in big Sigma (alphabet) MUST BE NO REPEATS
        f (float): parameter to promote forward progress through word
        s (float): parameter to decrease f_A if forward progress is not made
        u (float): parameter for useless events
        benign_events (list of str): events that are irrelevant if recorded and make no change to f_A output. All the trans matrices are identity
        actions_list (list of str): list of actions used in word
        objects_list (list of str): ordered list of objects in environment agent interacts with objects_list[0] first and then objects_list[1] and so forth
        f_reward (float): reward for progressing through WFA
        f_penalty (float): penalty (negative reward) for taking actions that have no progress
        finish_factor (float): reward for completing task = finish_factor*(1 - step_count/max_steps)
        env_size (int): environment will have grid dimensions of env_size by env_size
        render_mode (str): environment render mode 
    Returns:
        ord_obj_env (ordered_obj): instance of ordered_obj environment
    """

    alphabet = ["pickup ball", "pickup box", 
                "pickup key", "dropped ball", 
                "dropped box", "dropped key", 
                "useless"]

    WFA = word2WFA_max(word=word, alphabet=alphabet, benign_events=benign_events, 
                       f = f,  s = s, u = u)
    
    wfa_monitor = WFA_monitor(WFA=WFA, word=word )
    ord_obj_env = ordered_obj(wfa_monitor=wfa_monitor, 
                              actions_list=actions_list, 
                              objects_list=objects_list, 
                              size=env_size, 
                              f_reward=f_reward, 
                              f_penalty=f_penalty, 
                              finish_factor=finish_factor, 
                              render_mode=render_mode)
    return ord_obj_env

def create_multiroom_env( f_reward: Optional[float] = 10.0,
                           f_penalty: Optional[float] = 0.25,
                            finish_factor: Optional[float] = 10.0,
                              f: Optional[float] = 1.2, 
                               s: Optional[float] = 0.8, 
                                u: Optional[float] = 0.75,
                                render_mode: Optional[ str ] = "rgb_array"
                              ) -> WFA_TestEnv:
    """
    Creates multiroom with door key and box environment

    Args:
        f (float): parameter to promote forward progress through word
        s (float): parameter to decrease f_A if forward progress is not made
        u (float): parameter for useless events
        f_reward (float): reward for progressing through WFA
        f_penalty (float): penalty (negative reward) for taking actions that have no progress
        finish_factor (float): reward for completing task = finish_factor*(1 - step_count/max_steps)
        render_mode (str): environment render mode 
    Returns:
        multi_room_env (WFA_TestEnv) multi-room with box key and door
    """
    
    alphabet = ['pickup key', 'opened door', 
                'dropped key', 'closed door', 
                'pickup box','dropped box', 'useless']
    
    word = ['pickup key', 'opened door', 
                'dropped key', 'closed door', 
                'pickup box','dropped box']
    
    WFA = word2WFA_max(word=word,alphabet=alphabet, f=f, s=s, u=u)
    wfa_monitor = WFA_monitor(WFA=WFA)
    multi_room_env = WFA_TestEnv(WFA_monitor=wfa_monitor,
                                 f_reward=f_reward, 
                                 f_penalty=f_penalty,
                                 finish_factor=finish_factor, 
                                 render_mode=render_mode)
    return multi_room_env

if __name__ == "__main__":

    # ['pickup key', 'opened door', 
    #             'dropped key', 'closed door', 
    #             'pickup box', 'useless']
    spec_indicator = True
    if spec_indicator:
        # testing estimator
        direc       = r"C:\Users\nsmith3\Documents\GitHub\temporal_RLHF\BTS_models"

        need_new_EST = True  
        if need_new_EST:
            BT_SPEC_EST = create_BTS_EST()
            save_bts_est(bts_est=BT_SPEC_EST, folder_path=direc)
        

        filename = "bts_est_2025-05-22_12-29-45.pkl"  # Example; change to your actual file
        full_path = os.path.join(direc, filename)

        bts_est = load_bts_est(full_path)
        p0 = bts_est.learned_WFA.weight([])
        a = bts_est.learned_WFA.weight(['pickup key'])
        b = bts_est.learned_WFA.weight(['pickup key', 'opened door'])
        c = bts_est.learned_WFA.weight(['pickup key', 'opened door', 'dropped key'])
        d = bts_est.learned_WFA.weight(['pickup key', 'opened door', 'dropped key', 'closed door'])
        e = bts_est.learned_WFA.weight(['pickup key', 'opened door', 'dropped key', 'closed door', 'pickup box'])
        for index, word in enumerate( bts_est.words_list_string ):
            lweight     = bts_est.learned_WFA.weight(word)
            BT_score    = bts_est.weights_BT[index]
            lweight     = bts_est.learned_WFA.weight(word)
    else:
        # WFA_max function
        sigma_max = ['pickup key', 'opened door', 
                'dropped key', 'closed door', 
                'pickup box']
        alphabet = ['pickup key', 'opened door', 
                'dropped key', 'closed door', 
                'pickup box', 'useless', 'movement']
        benign_events = ['movement']

        wfa_max = word2WFA_max(word = sigma_max, alphabet=alphabet, benign_events=benign_events)

        print(f"t_0 is {wfa_max.initial}")
        print(f"trans_dict is {wfa_max.transitions}")
        print(f"t_f is {wfa_max.final}")