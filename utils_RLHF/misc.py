import sys
from datetime import datetime
import os
import pickle 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import choix 
import numpy as np
import random 
from splearn.datasets.data_sample import SplearnArray
from splearn.datasets.base import DataSample
from splearn.spectral import Spectral
from AUTOMATA.auto_funcs import spwfa2WFA, create_wfa_T1a

class BT_SPEC_Estimator:
    def __init__(self, WFA,
                         rank, 
                         max_length, num_pairs,  
                         num_hank_rows, 
                         num_hank_columns, 
                         scale_factor, alpha = 0.1,
                         train_words_num = None, 
                         show_WFA = False):
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
        self.WFA  = WFA
        self.rank = rank
        self.max_length = max_length
        self.num_pairs  = num_pairs
        self.num_hank_rows = num_hank_rows
        self.num_hank_columns = num_hank_columns
        self.scale_factor = scale_factor
        self.alpha = alpha
        self.train_words_num = train_words_num
        self.show_WFA = show_WFA
        self.num_letters = len( self.WFA.alphabet )

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

    def create_pairs(self):
        """
        Generate random strings of natural numbers with random length
        string length <= max_length
        mad value of number in string == num_letters - 1 
        since 0 is mapped to a letter
        """
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

    def pairs2weights(self):
        # learn weights
        weights_BT = choix.ilsr_pairwise(self.num_words_unique, 
                                        self.preference_list, 
                                        alpha=self.alpha)
        # print(weights_BT)

        # scale weights to be > 0 (does not affect probabilities)
        self.weights_BT = weights_BT + np.abs( np.min( weights_BT ) )

    def construct_splearnarray(self, 
                            scale_factor=10, 
                            Method = "top_score", 
                            top_scale=None):
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

    def repeat_construct(self,prefix_word_array, len_prefix, scale = None):
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

    def build_spec_est(self):
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
    BT_SPEC_EST.construct_splearnarray(Method=Method)
    BT_SPEC_EST.build_spec_est()
    return BT_SPEC_EST

if __name__ == "__main__":
    """
    ADD PROPORTIONALLY MORE FINISHED WORDS TO SPLEARN_ARRAY
    ADJUST REWARD STRUCTURE
    """
    # ['pickup key', 'opened door', 
    #             'dropped key', 'closed door', 
    #             'pickup box', 'useless']
    
    direc       = r"C:\Users\nsmith3\Documents\GitHub\temporal_RLHF\BTS_models"

    need_new_EST = False  
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