import random
from environments.Environment import Environment
import time
import numpy as np


class EnvironmentGrid2Dwithkeys(Environment):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

    def __init__(self, params):
        #print("we enter in EnvironmentGrid2Dwithkeys init")
        self.n_x = int(params['num_cells_grid2D'][0])
        self.n_y = int(params['num_cells_grid2D'][1])

        self.num_states = self.n_x * self.n_y
        self.states = [(x, y) for x in range(self.n_x) for y in range(self.n_y)]

        self.keys_position = [(random.randint(0, self.n_x - 1), random.randint(0, self.n_y - 1)) for i in range(2)]

        self.define_map()
        self.walls = []

        self.keys_taken = []
        assert self.num_states > 1, "Number of cells must be 2 or larger"

        self.num_actions = 4
        self.current_state = None

        self.terminal_state = (0, 0)  # arbitrary
        self.viewer = None

    def define_map(self):
        self.map = np.zeros((self.n_x, self.n_y))
        for key_pos in self.keys_position:
            self.map[key_pos[1]][key_pos[0]] = 5

    def is_done(self):

        if self.current_state == self.terminal_state and len(self.keys_taken)>=2:

            return True
        return False


    def step(self, action):
        """See documentation in the base class"""
        reward = 0
        # Decrease agent_cell by 1 if you go left, but only if you are not
        # in the left-most cell already
        if self.current_state in self.keys_position and self.current_state not in self.keys_taken:
                self.keys_taken.append(self.current_state)

        if action == EnvironmentGrid2Dwithkeys.LEFT:
            if self.current_state[0] > 0:
                self.current_state = (self.current_state[0]-1, self.current_state[1])

        if action == EnvironmentGrid2Dwithkeys.RIGHT:
            if self.current_state[0] < (self.n_x-1):
                self.current_state = (self.current_state[0]+1, self.current_state[1])

        if action == EnvironmentGrid2Dwithkeys.DOWN:
            if self.current_state[1] >0:
                self.current_state = (self.current_state[0], self.current_state[1] -1)

        if action == EnvironmentGrid2Dwithkeys.UP:
            if self.current_state[1] < (self.n_y-1):
                self.current_state = (self.current_state[0], self.current_state[1] +1)

        if self.is_done():
            # If you are in the terminal state, you've found the exit: reward!
            reward += 2000
        else:
            # Still wandering around: -1 penalty for each move
            reward += -1

        next_state = self.current_state

        return [next_state, reward, self.is_done()], self.keys_taken

    def state_to_indice(self, state):
        return self.states.index(state)

    def reset(self):
        """See documentation in the base class"""

        # Put agent at random position (but not in a terminal state)
        cell = (random.randint(0, self.n_x - 1), random.randint(0, self.n_y - 1))
        while cell == self.terminal_state:
            cell = (random.randint(0, self.n_x - 1), random.randint(0, self.n_y - 1))

        self.current_state = cell
        self.keys_taken = []

        # Return first observed state
        return self.current_state
