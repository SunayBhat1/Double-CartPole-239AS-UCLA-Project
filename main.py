from DQN_agent import DQN_agent
from AC_agent import AC_agent
from AC_2agent import AC_2agent
import numpy as np

directory = "ActorCritic_2Agent/"

args = {
        "gamma" : 0.99, # All
        
        "n_episode" : 5000, # All

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

Agent=DQN_agent(args)
Agent.load("dqn_end.pkl")
Agent.evaluate(True)

<<<<<<< HEAD
# Agent = AC_2agent(args)
# # Agent.load("ActorCritic/")
# Agent.run_training(directory,100)
# Agent.evaluate(directory,True)
=======
Agent = AC_2agent(args,'full')
Agent.load(directory)
Agent.run_training(directory,100)
Agent.evaluate(directory,True)
>>>>>>> 276aab9fc285c9d52c42ced016da2dde35b32732

