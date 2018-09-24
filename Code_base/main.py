from params import default_params
from environments.load_env import make_env
from agents.load_agent import make_agent
import matplotlib.pyplot as pyplot
import numpy as np
import argparse
import gc

np.random.seed(42)


def runEpisode(environment, agent, max_action_per_episode):
    """Run one episode.

    Integrate the environment-agent loop until a terminal state is reached,
    or a maximum number of actions.

    Args:
        environment (Environment): An environment
        agent (Agent): An agent
        max_action_per_episode (int): Maximum number of actions before an episode
            is terminated.
        render (bool) : to display learning
        debug (bool) : display debug information

    Returns:
        The return of the episode, i.e. the sum over all the rewards.
    """

    state = environment.reset()
    agent.get_infos_from_env(environment)

    action = agent.start(environment.state_to_indice(state), environment.get_possible_actions(state))

    is_done = False
    reward = 0.0
    total_reward = 0.0
    num_actions = 0
    way = []
    actions = []

    while (not is_done) and (num_actions < max_action_per_episode):
        info_state, key_taken = environment.step(action)
        state, reward, is_done = info_state
        action = agent.step(reward, environment.state_to_indice(state), environment.get_possible_actions(state))
        agent.num_actions = num_actions
        agent.key_taken = key_taken
        total_reward += reward
        way.append(state)

        actions.append(action)
        num_actions += 1

    return total_reward, way, actions


if __name__ == "__main__":

    params = default_params()
    env_name = params['env']
    agent_name = params['agent']
    environment = make_env(env_name, params)
    params['num_states'] = environment.num_states
    params['num_actions'] = environment.num_actions

    agent = make_agent(agent_name, params)

    y_rewards = []
    episodes = []
    # Run episodes
    try:
        environment.show_map()
        environment.show_map_elts()
    except:
        print("the env doens't have the show map method")
    for episode in range(params['num_training_episodes']):

        print("Episode ", episode, "starting.")

        total_reward, way, actions = runEpisode(environment, agent,
                                                params['max_action_per_episode'])

        y_rewards.append(total_reward)
        episodes.append(episode)
        print("Episode ", episode, " done. Total reward: ", total_reward)
        print("way: ", way)
        print("_______________________________")
        gc.collect()

        agent.update_post_episode(total_reward, len(way))


    # Plot the learning curve
    axes = pyplot.gca()
    axes.set_xlim([0, params['num_training_episodes']])
    axes.set_ylim([min(y_rewards) * 1.5, max(y_rewards) * 1.5])
    pyplot.scatter(episodes, y_rewards)
    granularity = 20
    averaged = []
    for i in range(granularity-1):
        len_average = len(y_rewards)/granularity
        averaged.append(np.mean(y_rewards[i*granularity:(i +1)* granularity]))
    pyplot.scatter(episodes, y_rewards)
    pyplot.show()
    pyplot.scatter(list(range(len(averaged))), averaged)
    pyplot.show()
