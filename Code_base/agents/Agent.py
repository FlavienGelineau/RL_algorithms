class Agent:
    """ The interface an agent should conform to
    """

    def __init__(self, params):
        """Initializes a new agent.
        Args:
            num_states (int): Number of possible states the agent can make
            num_actions (int): Number of possible action the agent can perform
        """

        self.num_states = params['num_states']
        self.num_actions = params['num_actions']

    def step(self, reward, state):
        """One step of the agent in the reinforcement learning loop.
            
        Usually, this function will do two things:
        1) Use the reward to update values
        2) Call the agent's policy to determine the next action, given 
           the state
           
        Args: 
            reward (float): The reward recieved for performing the previous
                action in the previous state
            state : The state at the current time step
        Returns:
            The action returned for the state at the current time step
        """
        raise NotImplementedError('subclasses must override agentStep()!')

    def policy(self, state):
        """The policy of an agent.
        
        In reinforcement learning, the policy returns an action, given the
        current state.
        
        Args: 
            state : The state the agent is in.
        Returns:
            The action (int) the agent performs.
        """
        raise NotImplementedError('subclasses must override agentPolicy()!')

    def get_infos_from_env(self, env):
        self.n_x = env.n_x
        self.n_y = env.n_y
        self.states = env.states
        self.map = env.map

    def start(self, initial_state, possible_actions):
        """Execute the first action with possibly untrained model.

        Initialize the states / actions met.
        """
        self.last_state = initial_state
        action = self.policy(initial_state, possible_actions)
        self.last_action = action

        return action
