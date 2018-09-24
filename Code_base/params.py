def default_params():
    """ Default parameters function

    Return :
        params_dict : dictionnary of parameters
    """

    params_dict = {

        ## Agent parameters
        'agent': 'SARSA',

        ## Environment parameters"
        'env': 'EnvironmentGrid2D',
        'num_cells_grid1D': 20,
        'num_cells_grid2D': (20, 20),

        ## Training parameters
        'num_training_episodes': 2500,
        'max_action_per_episode': 400

    }
    return params_dict

