import random
import numpy as np
from agents.Agent import Agent
from environments.EnvironmentGrid2Dwithkeys import EnvironmentGrid2Dwithkeys
import operator
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import gc

np.random.seed(42)


class MLPAgent(Agent):

    def __init__(self, params):
        Agent.__init__(self, params)

        # The 3 following args will be given by the env.
        self.n_x = None
        self.n_y = None
        self.states = None

        self.X_memory = []
        self.Y_memory = []
        self.ia = MLPRegressor(warm_start=True,
                               max_iter=200,
                               early_stopping=False,
                               hidden_layer_sizes=(20, 10, 5),
                               learning_rate_init=1 * 10 ** -3,
                               activation='identity')

        self.epsilon = 0.5
        self.gamma = 0.8
        self.alpha = 0.5

        self.last_action = None
        self.last_state = None
        self.is_action_possible = None
        self.n_max_action = params["max_action_per_episode"]

        self.first_fit = True
        self.key_taken = []

    def start(self, initial_state, possible_actions):
        """Execute the first action with possibly untrained model.

        Initialize the states / actions met.
        """
        self.last_state = initial_state
        action = self.policy(initial_state, possible_actions)
        self.last_action = action

        return action

    def get_agent_env(self, x, y):
        return []
        return [self.map[max(i, 0)][max(j, 0)] for i in range(x - 2, x + 1) for j in range(y - 2, y + 1)]

    def add_to_memory(self, X, Y):
        self.X_memory.extend([X])
        self.Y_memory.extend([Y])

    def step(self, reward, state, possible_action):
        """See documentation in the base class"""
        x_before, y_before = self.states[self.last_state]
        x, y = self.states[state]
        X = np.array(
            [x_before, y_before, self.last_action, len(self.key_taken)] + self.get_agent_env(x_before, y_before))

        action_choosen = self.policy(state, possible_action)
        if self.first_fit:
            Y = [reward]
        else:
            X_to_pred = np.array([[x, y, action_choosen, len(self.key_taken)] + self.get_agent_env(x, y),
                                  [x_before, y_before, self.last_action, len(self.key_taken)] + self.get_agent_env(
                                      x_before, y_before)
                                  ])
            preds = self.ia.predict(X_to_pred)
            adjusted_reward = reward * 1 if reward > 0 else reward
            Y = [(1 - self.alpha) * preds[1] + self.alpha * (adjusted_reward + self.gamma * preds[0])]

        self.add_to_memory(X, Y)

        if self.num_actions % 40 == 0:
            self.partial_X_memory = self.X_memory[-40:]
            self.partial_Y_memory = self.Y_memory[-40:]
            self.ia.fit(np.array(self.partial_X_memory), np.array(self.partial_Y_memory))
        if self.first_fit:
            self.ia.fit(np.array([X]), np.array([Y]))

            self.first_fit = False

        assert self.states[state] != self.states[self.last_state]

        self.last_action = action_choosen
        self.last_state = state
        gc.collect()

        return action_choosen

    def policy(self, state, possible_actions):
        """See documentation in the base class"""

        a = random.uniform(0, 1)
        if a < self.epsilon or self.first_fit == True:
            return random.choice(possible_actions)
        else:
            possible_rewards = {}

            for action in possible_actions:
                x, y = self.states[state]
                possible_rewards[action] = self.ia.predict(
                    np.array([x, y, action, len(self.key_taken)] + self.get_agent_env(x, y)).reshape(1, -1))

            reward_values = [possible_rewards[action] for action in possible_actions]
            if abs(max(reward_values) - min(reward_values)) < 0.1 * np.mean(reward_values):
                return random.choice(possible_actions)
            action = max(possible_rewards.items(), key=operator.itemgetter(1))[0]
        gc.collect()

        return action

    def update_post_episode(self, total_reward, n_epochs_last_episode):
        self.epsilon = max(self.epsilon * 0.99, 0.01)
        memory_len = n_epochs_last_episode * 4

        self.X_memory = self.X_memory[-memory_len:]
        self.Y_memory = self.Y_memory[-memory_len:]
        self.Y_memory[-n_epochs_last_episode:] = np.array(
            [elt[0] + (i / n_epochs_last_episode) * total_reward for i, elt in
             enumerate(self.Y_memory[-n_epochs_last_episode:])]).reshape(-1, 1)
        self.ia.fit(np.array(self.X_memory), np.array(self.Y_memory))

        gc.collect()
