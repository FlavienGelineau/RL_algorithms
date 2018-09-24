import random
from environments.Environment import Environment
import time
import numpy as np
import pandas as pd


class KeysandWalls(Environment):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

    def __init__(self, params, load_map_from_txt=True):

        if load_map_from_txt:
            self.map = self.load_map()
            self.n_x = len(self.map[0])
            self.n_y = len(self.map)
            self.states = [(x, y) for x in range(self.n_x) for y in range(self.n_y)]

            self.walls = [state for state in self.states if self.map[state[1]][state[0]] == 1]
            self.keys_position = [state for state in self.states if self.map[state[1]][state[0]] == 5]
        else:

            self.n_x = int(params['num_cells_grid2D'][0])
            self.n_y = int(params['num_cells_grid2D'][1])
            self.walls = [(random.randint(1, self.n_x - 1), random.randint(1, self.n_y - 1)) for i in range(20)]
            self.keys_position = [(random.randint(0, self.n_x - 1), random.randint(0, self.n_y - 1)) for i in range(2)]

        self.num_states = self.n_x * self.n_y

        self.keys_taken = []
        # Check if there are enough cells
        assert self.num_states > 1, "Number of cells must be 2 or larger"

        # Number of actions is always 2 (LEFT, RIGHT)
        self.num_actions = 4

        # Set to most right cell (will change in reset(...) anyway)
        self.current_state = None

        # end state
        self.terminal_state = (0, 0)  # arbitrary

        self.viewer = None

    def load_map(self):
        map = pd.read_csv('environments/map.txt').values
        map = [line[0].split(' ') for line in map]
        map = [[int(cell_type) for cell_type in line] for line in map]
        return np.array(map)

    def show_map_elts(self):
        print('-' * 50)
        print('-' * 50)
        print("key positions: {}".format(self.keys_position))
        print('-' * 50)
        print('-' * 50)

        print('-' * 50)
        print('-' * 50)
        print("wall positions: {}".format(self.walls))
        print('-' * 50)
        print('-' * 50)

    def show_map(self):
        map = np.zeros((self.n_x, self.n_y))
        print(self.n_x, self.n_y)
        for i, j in self.walls:
            map[j][i] = 1

        for i, j in self.keys_position:
            map[j][i] = 5
        print(map)

    def is_done(self):

        if self.current_state == self.terminal_state and len(self.keys_taken) > 1:
            return True
        return False

    def step(self, action):
        """See documentation in the base class.

        No need to test if the action is possible: impossible actions are forbidden in the agent."""
        reward = 0
        # Decrease agent_cell by 1 if you go left, but only if you are not
        # in the left-most cell already
        if self.current_state in self.keys_position and self.current_state not in self.keys_taken:
            self.keys_taken.append(self.current_state)
            reward += 100

        self.current_state = self.get_pos_after_action(self.current_state, action)

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
        # Put agent at random position (but not in a terminal state)
        cell = (random.randint(0, self.n_x - 1), random.randint(0, self.n_y - 1))
        while cell == self.terminal_state or cell in self.walls:
            cell = (random.randint(0, self.n_x - 1), random.randint(0, self.n_y - 1))
        self.current_state = cell
        self.keys_taken = []
        return self.current_state

    def is_action_possible(self, state, action):
        future_pos = self.get_pos_after_action(state, action)
        if future_pos in self.walls:
            return False
        if 0 <= future_pos[0] <= self.n_x - 1 and 0 <= future_pos[1] <= self.n_y - 1:
            return True

        return False

    def get_possible_actions(self, state):
        return [action for action in range(self.num_actions) if self.is_action_possible(state, action)]
