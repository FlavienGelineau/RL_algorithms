import sys
from environments.EnvironmentGrid1D import EnvironmentGrid1D
from environments.EnvironmentGrid2D import EnvironmentGrid2D
from environments.EnvironmentGrid2Dwithkeys import EnvironmentGrid2Dwithkeys
from environments.keysandwalls import KeysandWalls



def make_env(env_name, params):

	list_environments = {
	'EnvironmentGrid1D':EnvironmentGrid1D(params),
	'EnvironmentGrid2D': EnvironmentGrid2D(params),
	'EnvironmentGrid2Dwithkeys': EnvironmentGrid2Dwithkeys(params),
	'KeysandWalls': KeysandWalls(params)
	}

	try:

		env = list_environments[env_name]
	except:
		print('env_name not found')
		sys.exit()

	return env

