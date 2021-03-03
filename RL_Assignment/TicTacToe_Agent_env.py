from gym import spaces
import numpy as np
import random
from itertools import groupby
from itertools import product



class TicTacToe():

    def __init__(self):
        """initialise the board"""
        
        # initialise state as an array
        self.state = [np.nan for _ in range(9)]  # initialises the board position, can initialise to an array or matrix
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] # , can initialise to an array or matrix

        self.reset()


    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
        
        indices = np.reshape(np.array((range(9))),(3,3))
        row_ind = []
        col_ind = []
        for ind in range(indices.shape[0]):
            row_ind.append(indices[ind])
            col_ind.append(indices[:,ind])
            
        l_to_r_diag_ind = indices.diagonal()
        r_to_l_diag_ind = np.fliplr(indices).diagonal()
        
        row_col_diag_ind = [row_ind, col_ind, l_to_r_diag_ind, r_to_l_diag_ind]
        
        
        for ind in row_col_diag_ind:
            
            sum_check = curr_state[0] + curr_state[1] + curr_state[2]
            
            if sum_check == 15:
                return True
            else:
                return False
            

    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) == 0:
            return True, 'Tie'

        else:
            return False, 'Resume'


    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)


    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return (agent_actions, env_actions)



    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        curr_state[curr_action[0]] = curr_action[1]
        return curr_state


    def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. Hint: First, check the board position after
        agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""
        
        rwad = -1
        
        # RL agent plays its move
        next_state = self.state_transition(curr_state, curr_action)
        ter_state, result = self.is_terminal(next_state)
        
        
        if result == 'Win':
            rwad = 10
        elif result == 'Tie':
            rwad = 0
        
        # Environment player plays its move
        elif result == 'Resume':
            env_p  = random.choice(self.allowed_positions(next_state)) 
            env_v  = random.choice(self.allowed_values(next_state)[1])
        
            next_state[env_p]= env_v
            ter_state, result = self.is_terminal(next_state)                            
        
        elif result == 'Win':
            rwad = -10    
        
        else:
            rwad = 0
        print(result)
            
        return next_state, rwad, ter_state



    def reset(self):
        return self.state
