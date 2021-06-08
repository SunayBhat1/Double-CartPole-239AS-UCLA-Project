#from DQN_agent import DQN_agent
from DDQN_agent import DDQN_agent
from Rainbow_agent import Rainbow_agent
#from AC_agent import AC_agent
#from AC_2agent import AC_2agent
import numpy as np

directory = "./Rainbow/"

args = {
        "gamma" : 0.99, # All
        
        "n_episode" : 3000, # All

        "rand_angle" : 0.2, # All

        "mean_window" : 100, # All

        "horizon" : 200, # All

        "test_angles" : np.linspace(np.pi/8,-np.pi/8,100), #ALL

        "batch_size" : 256*2, # DQN, DDQN

        "hidden_dim" : 64, # DQN, DDQN

        "capacity" : 50000, # DQN, DDQN

        "max_episode" : 200, # DQN, DDQN, rainbow

        "min_eps" : 0.01, # DQN, DDQN

        "num_saved_episode" : 3, # ????

        "Load_DQN" : False, # DQN, DDQN

        'ac_hidden1_dim' : 128, # AC

        'ac_hidden2_dim': 256, # AC

        'alpha': 0.0001, # AC
    
        'seed': 100,

        'gpu': 0,

        'replay_start_size': 1600,

        'load_path': './Rainbow_result',

        'n_step_return': 3,
        
        'steps': 2 * 10 ** 6
    }

# Agent=DQN_agent(args)
# Agent.run_training("",100)
# Agent.load("dqn.pkl")
# Agent.evaluate(True)

# Agent = AC_2agent(args,'full')
# Agent.load(directory)
# Agent.run_training(directory,100)
# Agent.evaluate(directory,True)

#Agent=DDQN_agent(args)
#Agent.run_training("",3000)
Agent = Rainbow_agent(args)
#Agent.run_training(directory, True)
Agent.evaluate('./Plots/', True)
