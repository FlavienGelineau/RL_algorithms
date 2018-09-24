import random
from environments.Environment import Environment
import time
import numpy as np


class EnvironmentGrid2D(Environment):

    def __init__(self, params):

        self.n_x = int(params['num_cells_grid2D'][0])
        self.n_y = int(params['num_cells_grid2D'][1])

        self.num_states = self.n_x * self.n_y
        self.states = [(x, y) for x in range(self.n_x) for y in range(self.n_y)]
        self.map=np.zeros((self.n_x, self.n_y))
        self.walls = []
        # Check if there are enough cells
        assert self.num_states > 1, "Number of cells must be 2 or larger"

        self.num_actions = 4

        self.current_state = None

        # end state
        self.terminal_state = (0, 0)  # arbitrary

        self.viewer = None

    def step(self, action):
        """See documentation in the base class"""

        # Decrease agent_cell by 1 if you go left, but only if you are not
        # in the left-most cell already
        if action == EnvironmentGrid2D.LEFT:
            if self.current_state[0] > 0:
                self.current_state = (self.current_state[0]-1, self.current_state[1])


        # Increase agent_cell by 1 if you go right, but only if you are not
        # in the right-most cell already
        if action == EnvironmentGrid2D.RIGHT:
            if self.current_state[0] < (self.n_x-1):
                self.current_state = (self.current_state[0]+1, self.current_state[1])

        if action == EnvironmentGrid2D.DOWN:
            if self.current_state[1] >0:
                self.current_state = (self.current_state[0], self.current_state[1] -1)

        if action == EnvironmentGrid2D.UP:
            if self.current_state[1] < (self.n_y-1):
                self.current_state = (self.current_state[0], self.current_state[1] +1)


        is_done = self.current_state == self.terminal_state

        if is_done:
            # If you are in the terminal state, you've found the exit: reward!
            reward = 5000
        else:
            # Still wandering around: -1 penalty for each move
            reward = -1

        next_state = self.current_state

        return [next_state, reward, is_done], []
