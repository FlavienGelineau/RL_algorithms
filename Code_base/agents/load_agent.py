import sys
from agents.AgentRandom import AgentRandom
from agents.MonteCarlo import MonteCarlo
from agents.TD_Qlearning import Qlearning
from agents.SARSA import SARSA
from agents.AgentMLP import MLPAgent



def make_agent(agent_name, params):

	list_agent = {
	'MonteCarlo':MonteCarlo(params),
    'AgentRandom': AgentRandom(params),
    'TD_Qlearning':Qlearning(params),
    'SARSA': SARSA(params),
	'AgentMLP': MLPAgent(params)
	}

	try:
		agent = list_agent[agent_name]
	except:
		print('agent_name not found')
		sys.exit()

	return agent

