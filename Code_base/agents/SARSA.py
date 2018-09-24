import random
import numpy as np
from agents.Agent import Agent
from environments.EnvironmentGrid2Dwithkeys import EnvironmentGrid2Dwithkeys
import operator


class SARSA(Agent):

    def __init__(self, params):
        """See documentation in the base class"""
        Agent.__init__(self, params)
        self.n_x = None
        self.n_y = None
        self.states = None
        self.n_key_max = 5
        self.key_taken = []
        self.q = np.random.rand(self.num_states, self.num_actions, self.n_key_max)

        self.epsilon = 1

        self.alpha = 0.6
        self.gamma = 0.9999

        self.last_action = None
        self.last_state = None

    def step(self, reward, state, possible_actions):
        """See documentation in the base class"""

        action_choosen = self.policy(state, possible_actions)

        self.q[self.last_state][self.last_action][len(self.key_taken)] += self.alpha * (
                reward + self.gamma * self.q[state][action_choosen][len(self.key_taken)] -
                self.q[self.last_state][self.last_action][len(self.key_taken)])

        self.last_action = action_choosen
        self.last_state = state

        return action_choosen

    def policy(self, state, possible_actions):
        """See documentation in the base class"""
        a = random.uniform(0, 1)
        if a < self.epsilon:
            action = random.choice(possible_actions)
        else:
            possible_rewards = {action: self.q[state][action][len(self.key_taken)] for action in possible_actions}
            action = max(possible_rewards.items(), key=operator.itemgetter(1))[0]

        return action

    def update_post_episode(self, total_reward, len_way):
        self.epsilon = max(self.epsilon * 0.99, 0.01)
