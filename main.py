# from DQN_agent import DQN_agent
# from DDQN_agent import DDQN_agent
from AC_agent import AC_agent
from AC_2agent import AC_2agent
import numpy as np


args = {
        "n_episode" : 2000, # All
        # Default 0.99
        "gamma" : 0.99, # All
        # Default 0.2
        "rand_angle" : .05, # All
        # Default: 100
        "mean_window" : 100, # All
        # Default: 200
        "horizon" : 200, # All
        # Default: np.linspace(np.pi/8,-np.pi/8,100)
        "test_angles" : np.linspace(np.pi/8,-np.pi/8,100), #ALL
        # Default: 256*2
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
# # Agent.run_training(directory,500)
# # Agent.evaluate(directory,True)
# Agent.render_run()

# Actor Critic 2-Agent Full
directory = "ActorCritic_2Agent_Full/"
Agent = AC_2agent(args,'full')
Agent.load(directory,'')
# Agent.run_training(directory,1000)
# Agent.evaluate(directory,True)
Agent=DDQN_agent(args)
Agent.run_training("DDQN/",100)
