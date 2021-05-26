# from DQN_agent import DQN_agent
from AC_agent import AC_agent
import numpy as np

args = {
        "gamma" : 0.99, # All
        
        "n_episode" : 10000, # All

        "rand_angle" : np.pi/10, # All

        "mean_window" : 100, # All

        "horizon" : 200, # All

        "test_angles" : np.linspace(np.pi/8,-np.pi/8,100), #ALL

        "batch_size" : 256, # DQN, DDQN

        "hidden_dim" : 144, # DQN, DDQN

        "capacity" : 50000, # DQN, DDQN

        "max_episode" : 50, # DQN, DDQN

        "min_eps" : 0.1, # DQN, DDQN

        "num_saved_episode" : 3, # ????

        "Load_DQN" : False, # DQN, DDQN

        'ac_hidden1_dim' : 128, # AC

        'ac_hidden2_dim': 256, # AC

        'alpha': 0.0001 # AC
    }

# Agent=DQN_agent(args)
# Agent.run_training("")

Agent = AC_agent(args)
Agent.run_training("ActorCritic/",100)
Agent.evaluate("ActorCritic/",True)

