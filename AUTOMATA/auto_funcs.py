import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from automata.fa.dfa import DFA
from WA_package.weighted_automaton import WeightedAutomaton
from enum import Enum
import sympy as sp
import numpy as np 
# import builtins

# # Disable print
# builtins.print = lambda *args, **kwargs: None

class DFAMonitor:
    def __init__(self, dfa):
        "monitors discrete finite automaton"
        self.dfa = dfa
        self.current_state = dfa.initial_state
        self.history = []
        self.progress = False
        
        state_list = list(dfa.states)
        "list of states"
        
        self.num_states = len(state_list)
        
        self.states_dict = {state: index for index, state in enumerate(sorted(state_list))}
        "dictionary mapping between states and natural numbers"
  
    def step(self, symbol=None):
        """
        step function for dfa
        input - symbol
        """
        "assume no progress is made by default"
        self.progress = False
        if not symbol:
            pass
        else:
            """
            check for invalid symbols
            """
            if symbol not in self.dfa.input_symbols:
                print(f"Undefined symbol: {symbol}")
                raise ValueError(f"Invalid input symbol: {symbol}")
            elif symbol not in self.dfa.transitions[self.current_state]:
                raise ValueError(f"No transition from {self.current_state} on {symbol}")
            
            next_state = self.dfa.transitions[self.current_state][symbol]
            if not next_state == self.current_state:
                "transition has been made"
                self.progress = True
                # print(f"progress is {self.progress}")
            # print(f"{self.current_state} --[{symbol}]--> {next_state}")
            # print(f"state before transition is {self.current_state}")
            self.current_state = next_state
            # print(f"state post transition is {self.current_state}")
            self.history.append((self.current_state, symbol, next_state))

    def is_accepting(self):
        return self.current_state in self.dfa.final_states

    def state_label2array(self, state = None):
        """Maps state label to array representation eg [0, 1.0, 0]
        state - string label of automaton state
        obtains array rep of ucrrent stte if state = None
        """

        if not state:
            # current state is state evaluated post action
            state = self.current_state

        state_index = self.states_dict[state]

        # initialize auto state array to allow for input into neural network
        state_array = np.zeros(self.num_states, dtype=np.float32)

        # fill auto state array with right index
        state_array[state_index] = 1.0
        return state_array
    
    def reset(self):
        """Reset auto to initial state and clear history"""
        self.current_state = self.dfa.initial_state
        self.history = []

class WFA_monitor:
    def __init__(self, WFA, delta_thresh = 0.5, check_decrease=False, min_prob_thresh = 1e-30):
        "monitors weighted finite automaton"
        self.WFA = WFA
        "convert WFA arrays to numpy arrays from sympy arrays"
        self.initial = np.array( WFA.initial ).astype(np.float64)
        self.final   = np.array( WFA.final ).astype(np.float64)
        self.current_state = self.initial
        self.history = []
        self.progress = False
        self.trans_quality = True 
        self.delta_thresh = delta_thresh
        self.min_prob_thresh     = min_prob_thresh
        self.num_states = WFA.n
        self.prob_decrease = False
        self.prob_increase = False
        self.check_decrease= check_decrease
        self.prob = 1
        self.unstable_ind = False
        self.recent_event = None 
        self.transitions = {}
        for key in WFA.transitions:
            self.transitions[key] = np.array( WFA.transitions[key] ).astype(np.float64)
        
    
    def step(self, symbol=None):
        """
        step function for wfa
        input - symbol
        """
        "assume no progress is made by default"
        self.progress = False
        self.prob_increase = False
        if not symbol:
            # no transition made, all thats 
            # needed is to reset the progress attribute
            pass
        else:
            """
            check for invalid symbols
            """
            if symbol not in self.WFA.alphabet:
                print(f"Undefined symbol: {symbol}")
                raise ValueError(f"Invalid input symbol: {symbol}")
            self.recent_event = symbol
            next_state = self.current_state@self.transitions[symbol]
            
            # trans probabilites of current state and next
            current_prob = self.current_state@self.final
            next_prob    = next_state@self.final

            # current should be > next by markov
            delta_prob = current_prob - next_prob 

            if delta_prob <= -0.05:
            # likelihood magically increases 
                self.prob_increase = True 
            elif self.check_decrease:
            # check if there was a big likelihood increase
            # normalized likelihood change
                norm_delta_prob = np.abs( delta_prob/current_prob )
                if  norm_delta_prob > self.delta_thresh:
                    "transition has been made"
                    self.prob_decrease = True
                else:
                    self.prob_decrease = False
                    # print(f"progress is {self.progress}")
            # print(f"{self.current_state} --[{symbol}]--> {next_state}")
            # print(f"state before transition is {self.current_state}")

            self.current_state = next_state
            self.trans_prob()
            # print(f"state post transition is {self.current_state}")
            self.history.append((self.current_state, symbol, next_state))
    
    def trans_prob(self):
        """TEST THIS FUNCTION WHEN YOU RETURN"""
        self.prob = self.current_state@self.final
        if self.prob <= 0.1:
            self.trans_quality = False
            self.unstable_ind = np.isclose(self.prob, 0,atol=self.min_prob_thresh)
        else:
            self.trans_quality = True 
        # print(f"transition likelihood:{self.prob}")

    def reset(self):
        """Reset auto to initial state and clear history"""
        self.current_state = self.initial
        self.history = []


dfa_T1 = DFA(
    states={'q0', 'q1', 'q2', 'q3', 'q4', 'q5'},
    input_symbols={'pickup key', 'opened door', 
                   'dropped key', 'closed door', 
                   'pickup box', 'movement'},
    transitions={
        'q0':  {
                'pickup key': 'q1',        # Correct symbol → progress
                'opened door': 'q0',         # Wrong symbol → stay
                'dropped key': 'q0',
                'closed door': 'q0',
                'pickup box': 'q0', 
                'movement': 'q0'
                },
        'q1': {
                'pickup key': 'q1',        
                'opened door': 'q2',         # progress
                'dropped key': 'q1',
                'closed door': 'q1',
                'pickup box': 'q1', 
                'movement': 'q1'
                },
        'q2': {
                'pickup key': 'q2',        
                'opened door': 'q2',         
                'dropped key': 'q3',  #progress
                'closed door': 'q2',
                'pickup box': 'q2', 
                'movement': 'q2'
                },
        'q3': {
                'pickup key': 'q3',        # Correct symbol → progress
                'opened door': 'q3',         # Wrong symbol → stay
                'dropped key': 'q3',
                'closed door': 'q4',
                'pickup box': 'q3', 
                'movement': 'q3',
                },
        'q4': {
                'pickup key': 'q4',        # Correct symbol → progress
                'opened door': 'q4',         # Wrong symbol → stay
                'dropped key': 'q4',
                'closed door': 'q4',
                'pickup box': 'q5', 
                'movement': 'q4'
                },
        'q5': {
            'pickup key': 'q5',
            'opened door': 'q5',
            'dropped key': 'q5',
            'closed door': 'q5',
            'pickup box': 'q5', 
            'movement': 'q5'
        }
    },
    initial_state='q0',
    final_states={'q5'}
)

def frac(a, b):
    return sp.Rational(a, b)

def simple_wfa() -> WeightedAutomaton:
    initial = sp.Matrix([[8, 8, 4, 6]])
    transitions = {'a': sp.Matrix(
        [[0, -1, 0, 0],
         [1, 2, 0, 0],
         [0, 0, 0, -1],
         [0, 0, 1, 2]]), 'b': sp.Matrix(
        [[0, 0, -frac(1, 2), 0],
         [0, 0, 0, -frac(1, 2)],
         [1, 0, frac(3, 2), 0],
         [0, 1, 0, frac(3, 2)]])}
    final = sp.Matrix([[1], [0], [0], [0]])

    return WeightedAutomaton(n=4,
                             alphabet=['a', 'b'],
                             initial=initial,
                             transitions=transitions,
                             final=final)

def create_wfa_T1(f, s, u=0.8):
    """
    Function to create wfa representation of task 1
    INPUTS 
    f - weight assigned for forward progress
    s - weight assigned for stationary progres
    u - weight assigned for useless actions e.g. pickup an item when nothing is in front
    OUTPUTS
    wfa_T1 - WFA object representing task 1
    """
    num_letters = 7
    num_states  = 6

    initial_array = sp.Matrix([[1, 0, 0, 0, 0, 0]])
    final_array   = sp.Matrix.ones(6,1)
    transition_dictionary = {}
    
    keys = {}
    symbols = ['pickup key', 'opened door', 
                'dropped key', 'closed door', 
                'pickup box', 'movement', 'useless']
    for index in np.arange(num_letters-2):
        progress_matrix   = sp.Matrix.zeros(num_states, num_states)
        progress_matrix[index, index+1] = f

        adjustment_matrix = sp.Matrix.zeros(num_states, num_states)
        adjustment_matrix[index,index] = s

        transition_matrix = s*sp.eye(num_states) + progress_matrix - adjustment_matrix
        symbol = symbols[index]
        transition_dictionary[symbol] = transition_matrix
    transition_dictionary["movement"] = sp.Matrix.eye(num_states)
    transition_dictionary['useless']  = u*sp.Matrix.eye(num_states)

    wfa_T1 = WeightedAutomaton(n=num_states,alphabet=symbols, 
                                initial=initial_array, 
                                transitions=transition_dictionary, 
                                final=final_array)
    return wfa_T1

def create_wfa_T1a(f, s, u=0.8):
    """
    Function to create wfa representation of task 1 
    with movement removed from alphabet
    INPUTS 
    f - weight assigned for forward progress
    s - weight assigned for stationary progres
    u - weight assigned for useless actions e.g. pickup an item when nothing is in front
    OUTPUTS
    wfa_T1a - WFA object representing task 1
    """
    num_letters = 6
    num_states  = 6

    initial_array = sp.Matrix([[1, 0, 0, 0, 0, 0]])
    final_array   = sp.Matrix.ones(6,1)
    transition_dictionary = {}
    
    symbols = ['pickup key', 'opened door', 
                'dropped key', 'closed door', 
                'pickup box', 'useless']
    for index in np.arange(num_letters-1):
        progress_matrix   = sp.Matrix.zeros(num_states, num_states)
        progress_matrix[index, index+1] = f

        adjustment_matrix = sp.Matrix.zeros(num_states, num_states)
        adjustment_matrix[index,index] = s

        transition_matrix = s*sp.eye(num_states) + progress_matrix - adjustment_matrix
        symbol = symbols[index]
        transition_dictionary[symbol] = transition_matrix
    transition_dictionary['useless']  = u*sp.Matrix.eye(num_states)

    wfa_T1a = WeightedAutomaton(n=num_states,alphabet=symbols, 
                                initial=initial_array, 
                                transitions=transition_dictionary, 
                                final=final_array)
    return wfa_T1a

def spwfa2WFA(wfa, alphabet=None) -> WeightedAutomaton:
    """
    Maps splearn wfa to WFA object defined in WFA_package that is much easier to work with
    IMPORTANT !!!
    wfa and alphabet must be in accordance
    ASSUME the ith transition array in the list provided by wfa.transitions is 
    the transition array for the ith letter in alphabet list provided as input
    INPUTS
    wfa - automaton object from splearn
    alphabet - ordered list of letters for wfa
    OUTPUTS
    WFA - wfa object from WFA_package
    """

    trans_list  = wfa.transitions
    num_letters = len( trans_list )

    if alphabet is None:
        alphabet = np.arange(num_letters)

    # call for arrays from splearn wfa
    num_states  = wfa.initial.shape[0]
    initial     = sp.Matrix( wfa.initial.reshape( (1,num_states) ) )
    final       = sp.Matrix( wfa.final.reshape( (num_states, 1) ) )
    trans_list  = wfa.transitions
    transitions = {}

    for letter_index in np.arange(num_letters):
        transitions[ alphabet[letter_index] ] = sp.Matrix( trans_list[letter_index] )

    WFA = WeightedAutomaton(n=num_states, alphabet=alphabet,
                            initial=initial, final=final, 
                            transitions=transitions)

    return WFA


if __name__ == "__main__":
    # monitor_T1 = DFAMonitor(dfa_T1)

    # # Simulate input symbols arriving one by one
    # monitor_T1.step('pickup key')
    # monitor_T1.step('opened door')
    # monitor_T1.step('dropped key')

    # print("✅ Accepting?" if monitor_T1.is_accepting() else "❌ Not accepting")
    # print("Current state:", monitor_T1.current_state)
    # print("History:", monitor_T1.history)
    f = 0.95
    s = 0.01
    wfa_T1 = create_wfa_T1(f=f, s=s)
    initial_T1 = wfa_T1.initial
    final_T1   = wfa_T1.final
    trans_T1   = wfa_T1.transitions
    for key in trans_T1:
        trans = np.array( trans_T1[key] ).astype( np.float64 )

    test_good_word = [ 'pickup key', 'opened door', 
                    'dropped key', 'closed door', 
                    'pickup box', 'movement']
    
    test_good_word2 = [ 'pickup key', 
                      'opened door', 
                      'movement', 
                      'movement', 
                      'movement', 
                    'dropped key', 
                    'closed door', 
                    'pickup box', 
                    'movement']
    
    test_mediocre_word = [ 'pickup key', 'opened door', 'closed door', 'opened door',
                    'dropped key', 'closed door', 
                    'pickup box', 'movement']
    good_weight = wfa_T1.weight(test_good_word2)
    mediocre_weight = wfa_T1.weight(test_mediocre_word)
    print(f"{good_weight}")