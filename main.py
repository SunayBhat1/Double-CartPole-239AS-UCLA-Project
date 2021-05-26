from DQN_agent import DQN_agent
#from AC_agent import AC_agent

import numpy as np

args = {
        "gamma" : 0.99,
        
        "n_episode" : 15000,

        "batch_size" : 256,

        "hidden_dim" : 144,

        "capacity" : 50000,

        "max_episode" : 50,

        "min_eps" : 0.1,

        "num_saved_episode" : 3,

        "Load_DQN" : False,

        "horizon" : 200,

        "rand_angle" : np.pi/8,

        "mean_window" : 100,
        
        "test_angles" : np.linspace(np.pi/10,-np.pi/10,100),

        'ac_hidden1_dim' : 128,

        'ac_hidden2_dim': 256

    }
Agent=DQN_agent(args)
Agent.run_training("")



