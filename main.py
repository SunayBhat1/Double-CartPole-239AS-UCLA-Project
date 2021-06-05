# from DQN_agent import DQN_agent
from DDQN_agent import DDQN_agent
from AC_agent import AC_agent
from AC_2agent import AC_2agent
import numpy as np


args = {
        "n_episode" : 35000, # All
        # Default 0.99
        "gamma" : 0.9, # All
        # Default 0.2
        "rand_angle" : 0.1, # All
        # Default: 100
        "mean_window" : 100, # All
        # Default: 200
        "horizon" : 200, # All
        # Default: np.linspace(np.pi/8,-np.pi/8,100)
        "test_angles" : np.linspace(np.pi/8,-np.pi/8,100), #ALL
        # Default: 256*2
        "batch_size" : 256*2, # DQN, DDQN
        # Default: 64
        "hidden_dim" : 64, # DQN, DDQN
        # Default: 50000
        "capacity" : 50000, # DQN, DDQN
        # Default: 50
        "max_episode" : 50, # DQN, DDQN
        # Default: 0.01
        "min_eps" : 0.01, # DQN, DDQN
        # Default: 128
        'ac_hidden1_dim' : 128, # AC
        # Default: 256
        'ac_hidden2_dim': 256, # AC
        # Default:  0.0001
        'alpha': 0.0001 # AC
    }

# DQN Agent
# directory = "DQN/"
# Agent=DQN_agent(args)
# Agent.run_training("",100)
# Agent.load("dqn.pkl")
# Agent.evaluate(True)

# DDQN Agent
# directory = "DDQN/"
# Agent=DDQN_agent(args)
# Agent.run_training("",100)
# Agent.load(directory + 'DDQN_Q1.pt',file_ext='')
# Agent.evaluate(directory,True)
# Agent.render_run(directory,save_video=False)

# Actor Critic Agent
directory = "ActorCritic/"
Agent=AC_agent(args)
Agent.load(directory,'')
# Agent.run_training(directory,500)
# Agent.evaluate(directory,True)
Agent.render_run(directory,True,1)

# Actor Critic 2-Agent Full
# directory = "ActorCritic_2Agent_Full/"
# Agent = AC_2agent(args,'full')
# Agent.load(directory,'')
# Agent.run_training(directory,1000)
# Agent.evaluate(directory,True)
# Agent.render_run(10)

# Actor Critic 2-Agent Partial
# directory = "ActorCritic_2Agent_Partial/"
# Agent = AC_2agent(args,'partial')
# # Agent.load(directory)
# Agent.run_training(directory,1000)
# Agent.evaluate(directory,True)
