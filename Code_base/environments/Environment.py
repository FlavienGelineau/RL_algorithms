import random


class Environment:
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

    def __init__(self):
        pass

    def step(self, action):
        """Project environment one step into the future.
        
         Given an action, compute the reward and next state, and whether next_state is terminal
            
         Args:
             action : The action the agent is performing
             
         Returns:
            next_state : an observation of the next state
            reward : the reward the agent receives for performing action in the current state
            is_done = if the next_state is a terminal state
        """

        raise NotImplementedError('subclasses must override step()!')

    def render(self):
        """ Display the environment
        """

        raise NotImplementedError('subclasses must override render()!')

    def close(self):
        """ Close the environment
        """
        raise NotImplementedError('subclasses must override close()!')

    def state_to_indice(self, state):
        return self.states.index(state)

    def reset(self):
        """See documentation in the base class"""

        # Put agent at random position (but not in a terminal state)
        cell = (random.randint(0, self.n_x - 1), random.randint(0, self.n_y - 1))
        while cell == self.terminal_state:
            cell = (random.randint(0, self.n_x - 1), random.randint(0, self.n_y - 1))
        self.current_state = cell

        return self.current_state

    def get_pos_after_action(self, state, action):
        if action == self.LEFT:
            return (state[0] - 1, state[1])
        if action == self.RIGHT:
            return (state[0] + 1, state[1])
        if action == self.DOWN:
            return (state[0], state[1] - 1)
        if action == self.UP:
            return (state[0], state[1] + 1)

    def is_action_possible(self, state, action):
        future_pos = self.get_pos_after_action(state, action)
        if future_pos in self.walls:
            return False
        if 0 <= future_pos[0] <= self.n_x - 1 and 0 <= future_pos[1] <= self.n_y - 1:
            return True

        return False

    def get_possible_actions(self, state):
        return [action for action in range(self.num_actions) if self.is_action_possible(state, action)]
