from automata.fa.dfa import DFA
from WA_package.weighted_automaton import WeightedAutomaton
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

def create_wfa_T1(f, s):
    """
    Function to create wfa representation of task 1
    INPUTS 
    f - weight assigned for forward progress
    s - weight assigned for stationary progres
    OUTPUTS
    wfa_T1 - WFA object representing task 1
    """

if __name__ == "__main__":
    monitor_T1 = DFAMonitor(dfa_T1)

    # Simulate input symbols arriving one by one
    monitor_T1.step('pickup key')
    monitor_T1.step('opened door')
    monitor_T1.step('dropped key')

    print("✅ Accepting?" if monitor_T1.is_accepting() else "❌ Not accepting")
    print("Current state:", monitor_T1.current_state)
    print("History:", monitor_T1.history)
