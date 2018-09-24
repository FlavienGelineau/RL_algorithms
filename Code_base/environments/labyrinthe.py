__author__ = "Florence Carton"
__credits__ = ["Florence Carton", "Freek Stulp", "Antonin Raffin"]

import random
from environments.Environment import Environment
import time
import numpy as np


class labyrinthe(Environment):
    LEFT = 0
    RIGHT = 1
    DOWN = 2
    UP = 3

    def __init__(self, params):
        self.num_states_x = int(params['num_cells_grid2D_x'])
        self.num_states_y = int(params['num_cells_grid2D_y'])
        self.num_states = [self.num_states_x, self.num_states_y]

        assert (self.num_states_x > 1 and self.num_states_y > 1)

        self.num_actions = 4
        self.current_state = [9, 4]
        self.terminal_state = [0, 0]

        self.viewer = None

        # we define walls to build the labyrinthe

        self.walls = [random.randint(0, self.num_states_x - 1), random.randint(0, self.num_states_y - 1)]

    def step(self, action):
        """See documentation in the base class"""

        if action == labyrinthe.LEFT:
            # faire une boucle c'est peut-etre mieux
            if self.current_state[0] > 0:
                self.current_state = [self.current_state[0] - 1, self.current_state[1]]

        # Increase agent_cell by 1 if you go right, but only if you are not
        # in the right-most cell already
        if action == labyrinthe.RIGHT:
            if self.current_state[0] < self.num_states_x - 1:
                self.current_state = [self.current_state[0] + 1, self.current_state[1]]

        # if you go up, agent_cell 

        if action == labyrinthe.UP:
            if self.current_state[1] > 0:
                self.current_state = [self.current_state[0], self.current_state[1] - 1]

        if action == labyrinthe.DOWN:
            if self.current_state[1] < self.num_states_y - 1:
                self.current_state = [self.current_state[0], self.current_state[1] + 1]

        is_done = self.current_state == self.terminal_state

        if is_done:
            # If you are in the terminal state, you've found the exit: reward!
            reward = 100
        else:
            # Still wandering around: -1 penalty for each move
            reward = -1

        next_state = self.current_state

        return [next_state, reward, is_done]

    def print_map(self):
        map = np.eye(self.num_states_x, self.num_sates_y)
        print(map)

    # faire dans la mtrice si on monte si on tombe sur 1 alors stop sinon on continue

    def action_possible(self, state, action):
        action_next = self.step(self, action)
        if action_next in self.walls:
            return False
        if 0 <= action_next <= self.num_states_x - 1 and 0 <= action_next <= self.num_states - y - 1:
            return True

    def reset(self):
        """See documentation in the base class"""

        # Put agent at random position (but not in a terminal state)
        cell = [random.randint(0, self.num_states_x - 1), random.randint(0, self.num_states_y - 1)]
        while cell == self.terminal_state:
            cell = [random.randint(0, self.num_states_x - 1), random.randint(0, self.num_states_y - 1)]
        self.current_state = cell

        # Return first observed state
        return self.current_state

    def render(self):
        screen_width = 500
        screen_height = 75
        cell_width = screen_width / self.num_states_x
        cell_height = screen_height / self.num_states_y

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l, r, t, b = 0, cell_width, cell_height, 0
            cell = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])

            cell.set_color(0, 0, 1)  # blue for current state

            end = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            end.set_color(1, 0, 0)  # red for end state (note : end state is 0)
            self.viewer.add_geom(end)
            self.celltrans = rendering.Transform()

            cell.add_attr(self.celltrans)

            self.viewer.add_geom(cell)

        if self.current_state is None: return None

        state = self.current_state
        cellx = state[0] * cell_width

        celly = state[1] * cell_height
        self.celltrans.set_translation(cellx, celly)
        i = 0
        j = 0
        for (i == j):
            l, r, t, b = 0, cell_width, cell_height, 0
            wall = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            wall.set_color(0, 1, 0)  # green for wal

            wallx = state[0] * i * cell_width
            wally = state[1] * j * cell_height
            i = i + 1
            j = j + 1

        time.sleep(0.1)

        return self.viewer.render()

    def close(self):
        if self.viewer: self.viewer.close()
