import random
import numpy as np
from agents.Agent import Agent
import operator

class MonteCarlo(Agent):

    def __init__(self, params):
        """See documentation in the base class"""
        Agent.__init__(self, params)
        self.q = np.random.rand(self.num_states, self.num_actions)
        self.returns = {(state, action):[] for state in range(self.num_states) for action in range(self.num_actions)}
        self.epsilon = 1
        self.tuple_state_agent_met=[]


    def step(self, reward, state, possible_actions):
        """See documentation in the base class"""
        action = self.policy(state, possible_actions)

        self.tuple_state_agent_met.append((state, action))

        return action

    def policy(self, state, possible_actions):
        """See documentation in the base class"""
        a=random.uniform(0, 1)
        if a<self.epsilon:
            action = random.choice(possible_actions)
        else:
            possible_rewards = {action: self.q[state][action] for action in possible_actions}
            action = max(possible_rewards.items(), key=operator.itemgetter(1))[0]

        return action


    def update_post_episode(self, total_reward, len_way):
        for state_seen, action_done in self.tuple_state_agent_met:
            self.returns[(state_seen, action_done)].append(total_reward)
        for state_seen, action_done in self.tuple_state_agent_met:
            self.q[state_seen][action_done]=np.mean(self.returns[(state_seen, action_done)])
        for state_seen, action_done in self.tuple_state_agent_met:
            self.returns[(state_seen, action_done)]=self.returns[(state_seen, action_done)][-2000:]
        self.tuple_state_agent_met=[]
        self.epsilon = max(self.epsilon*0.99,0.001)
