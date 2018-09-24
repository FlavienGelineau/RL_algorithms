import random
from environments.Environment import Environment
import time

class EnvironmentGrid1D(Environment):

    LEFT = 0
    RIGHT = 1

    def __init__(self,params):

        self.num_states = int(params['num_cells_grid1D'])

        # Check if there are enough cells 
        assert self.num_states>1, "Number of cells must be 2 or larger"
        
        # Number of actions is always 2 (LEFT, RIGHT)
        self.num_actions = 2 
  
        # Set to most right cell (will change in reset(...) anyway)
        self.current_state  =  None

        # end state
        self.terminal_state = 0 # arbitrary

        self.viewer = None


    def step(self, action):
        """See documentation in the base class"""
        
        # Decrease agent_cell by 1 if you go left, but only if you are not
        # in the left-most cell already
        if action==EnvironmentGrid1D.LEFT:
            if self.current_state>0:
                self.current_state -= 1

        # Increase agent_cell by 1 if you go right, but only if you are not
        # in the right-most cell already
        if action==EnvironmentGrid1D.RIGHT:     
            if self.current_state<(self.num_states-1):
                self.current_state += 1

        is_done = self.current_state == self.terminal_state
       
        if is_done:
            # If you are in the terminal state, you've found the exit: reward!
            reward = 100
        else:
            # Still wandering around: -1 penalty for each move
            reward = -1
  
        next_state = self.current_state
      
        return [next_state,reward,is_done]
        

    def reset(self):
        """See documentation in the base class"""

        # Put agent at random position (but not in a terminal state)
        cell = random.randint(0,self.num_states-1)
        while cell == self.terminal_state:
            cell = random.randint(0,self.num_states-1)
        self.current_state = cell
        
        # Return first observed state
        return self.current_state
