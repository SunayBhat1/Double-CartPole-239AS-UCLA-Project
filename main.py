# from DQN_agent import DQN_agent
# from DDQN_agent import DDQN_agent
from AC_agent import AC_agent
from AC_2agent import AC_2agent
import numpy as np


args = {
        "n_episode" : 10000, # All
        
        "gamma" : 0.99, # All

        "rand_angle" : .1, # All

        "mean_window" : 100, # All

        "horizon" : 200, # All

        "test_angles" : np.linspace(np.pi/8,-np.pi/8,100), #ALL

        "batch_size" : 256*2, # DQN, DDQN

        "hidden_dim" : 64, # DQN, DDQN

        "capacity" : 50000, # DQN, DDQN

        "max_episode" : 50, # DQN, DDQN

        "min_eps" : 0.01, # DQN, DDQN

        "num_saved_episode" : 3, # ????

        "Load_DQN" : False, # DQN, DDQN

        'ac_hidden1_dim' : 128, # AC

        'ac_hidden2_dim': 256, # AC

        'alpha': 0.0001 # AC
    }

# DQN Agent
# directory = "DQN/"
# Agent=DQN_agent(args)
# Agent.run_training("",100)
# Agent.load("dqn.pkl")
# Agent.evaluate(True)

# Actor Critic Agent
# directory = "ActorCritic/"
# Agent=AC_agent(args)
# Agent.load(directory)
# Agent.run_training(directory,500)
# Agent.evaluate(directory,True)
# Agent.render_run()

# Actor Critic 2-Agent Full
directory = "ActorCritic_2Agent_Full/"
Agent = AC_2agent(args,'full')
# Agent.load(directory)
Agent.run_training(directory,1000)
Agent.evaluate(directory,True)
Agent.render_run()

# Actor Critic 2-Agent Partial
# directory = "ActorCritic_2Agent_Partial/"
# Agent = AC_2agent(args,'partial')
# # Agent.load(directory)
# Agent.run_training(directory,1000)
# Agent.evaluate(directory,True)