from agents.Agent import Agent
from environments.EnvironmentGrid1D import EnvironmentGrid1D
from environments.EnvironmentGrid2D import EnvironmentGrid2D
import numpy as np
import random


class AgentMonteCarlo(Agent):

    def __init__(self, params):
        """See documentation in the base class"""
        Agent.__init__(self, params)
        self.final_state = EnvironmentGrid2D(params).terminal_state


    def start(self, initial_state):
        """See documentation in the base class"""

        return self.policy(initial_state)

    def step(self, reward, state):
        """See documentation in the base class"""

        action = self.policy(state)
        test_reward = self.state == self.final_state
        if test_reward:
            reward = 100
        else:
            reward = -1
        return action

    def random_policy(self, state):
        return [random.randint(0, self.num_actions - 1)]

    def matrix(self, state):
        # On commence par établir une matrice sur num_it itérations
        # num_states_x = 500
        # num_states_y = 75
        M_final = np.zeros((75, 500))
        departure_state = [4, 9]

        # On récure sur num_it itérations
        for i in range(2):
            M_int = np.zeros((75, 500))
            reward_tot = 0
            reward = 0
            while (reward != 100):
                M_int[departure_state[0], departure_state[1]] = 1
                action = self.random_policy(state)
                test_reward = state == [0, 0]
                if (test_reward):
                    reward = 100
                else:
                    reward = -1
                reward_tot = reward_tot + reward

                if action == 0:
                    M_int[state[0], max(state[1] - 1, 0)] = 1
                elif action == 1:
                    M_int[state[0], min(state[1] + 1, 499)] = 1
                elif action == 2:
                    M_int[min(state[0] + 1, 74), state[1]] = 1
                else:
                    M_int[max(state[0], 0), state[1]] = 1
            M_final = M_final + M_int * reward

        # On renvoie la matrice coefficientée
        return M_final

    def policy(self, state):

        Matrix = self.matrix(state)
        Left = Matrix[state[0]][max(state[1] - 1, 0)]
        Right = Matrix[state[0]][min(state[1] - 1, num_states_y - 1)]
        Up = Matrix[max(state[0] - 1, 0)][state[1]]
        Down = Matrix[min(state[0] + 1, num_states_x - 1)][state[1]]

        # Attention : Il ne peut pas avancer en diagonale !
        if max(Left, Right) < max(Up, Down):
            if Left < Right:
                return [state[0], min(state[1] - 1, num_states_y - 1)]
            else:
                return [state[0], max(state[1] - 1, 0)]
        else:
            if Up < Down:
                return [min(state[0] + 1, num_states_x - 1), state[1]]
            else:
                return [max(state[0] - 1, 0), state[1]]
