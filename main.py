#from DQN_agent import DQN_agent
from DDQN_agent import DDQN_agent
from Rainbow_agent import Rainbow_agent
from AC_agent import AC_agent
from AC_2agent import AC_2agent
import numpy as np
import matplotlib.pyplot as plt

args = {
        "n_episode" : 3000, # All
        # Default 0.99
        "gamma" : 0.99, # All
        
        "n_episode" : 3000, # All

        "rand_angle" : 0.2, # All

        "mean_window" : 100, # All
        # Default: 200
        "horizon" : 200, # All
        # Default: np.linspace(np.pi/8,-np.pi/8,100)
        "test_angles" : np.linspace(np.pi/8,-np.pi/8,100), #ALL
        # Default: 256*2
        "batch_size" : 256*2, # DQN, DDQN
        # Default: 64
        "hidden_dim" : 144, # DQN, DDQN
        # Default: 50000
        "capacity" : 50000, # DQN, DDQN
        # Default: 50
        "max_episode" : 200, # DQN, DDQN
        # Default: 0.01
        "min_eps" : 0.01, # DQN, DDQN
        # Default: 128
        'ac_hidden1_dim' : 128, # AC
        # Default: 256
        'ac_hidden2_dim': 256, # AC
        # Default: 256
        'ac_hidden3_dim': 512, # AC
        # Default:  0.0001
        'alpha': 0.0001, # AC

        'n_step_return': 5,
        'replay_start_size': 42,

        'load_path': '/Rainbow/model.pt',

        'steps': 100,
    
        'seed': 100,

        'gpu': 0,

        'replay_start_size': 1600,

        'load_path': './Rainbow_result',

        'n_step_return': 3,
        
        'steps': 2 * 10 ** 6
    }

# # Rainbow Agent
# Agent=Rainbow_agent(args)


# # DQN Agent
# directory = "DQN/"
# Agent=DQN_agent(args)
# # Agent.run_training("",100)
# Agent.load('DQN/DQN_5_20_0010.pt')
# vec_dqn = Agent.evaluate(True)
# np.save('Results/DQN_evaluate.npy',vec_dqn)

# args['hidden_dim'] = 64
# # # DDQN Agent
# directory = "DDQN/"
# Agent=DDQN_agent(args)
# # # Agent.run_training("",100)
# Agent.load(directory + 'DDQN_Q1.pt',file_ext='')
# vec_ddqn = Agent.evaluate(directory,True)
# # Agent.render_run(directory,save_video=False)
# np.save('Results/DDQN_evaluate.npy',vec_ddqn)

# # Actor Critic Agent
# directory = "ActorCritic/"
# Agent=AC_agent(args)
# Agent.load(directory,'')
# # Agent.run_training(directory,500)
# vec_ac = Agent.evaluate(directory,True)
# # Agent.render_run(directory,True,1)
# np.save('Results/AC_evaluate.npy',vec_ac)


# Plot Results
fig7, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(6,3.5), dpi= 130, facecolor='w', edgecolor='k')
fig7.suptitle('Start Angle vs Episode Length',fontweight='bold',fontsize = 14)
ax1.plot(args['test_angles']*180/np.pi,vec_dqn,c='g')
ax2.plot(args['test_angles']*180/np.pi,vec_ac,c='b')
ax3.plot(args['test_angles']*180/np.pi,vec_ddqn,c='r')
ax4.plot(args['test_angles']*180/np.pi,vec_rain,c='k')
ax1.set_ylabel("Seconds",fontweight='bold',fontsize = 10)
ax1.set_xlabel("Degrees",fontweight='bold',fontsize = 10)
ax2.set_ylabel("Seconds",fontweight='bold',fontsize = 10)
ax2.set_xlabel("Degrees",fontweight='bold',fontsize = 10)
ax3.set_ylabel("Seconds",fontweight='bold',fontsize = 10)
ax3.set_xlabel("Degrees",fontweight='bold',fontsize = 10)
ax4.set_ylabel("Seconds",fontweight='bold',fontsize = 10)
ax4.set_xlabel("Degrees",fontweight='bold',fontsize = 10)
ax1.legend(['DQN'])
ax2.legend(['AC'])
ax3.legend(['DDQN'])
ax4.legend(['Rainbow'])
fig7.savefig('Results_All.png')

mask = abs(args['test_angles'] * 180/np.pi) < 12
masked_results_rain = np.ma.array(vec_rain,mask = ~mask)
masked_results_ac = np.ma.array(vec_ac,mask = ~mask)
masked_results_ddqn = np.ma.array(vec_ddqn,mask = ~mask)
masked_results_dqn = np.ma.array(vec_dqn,mask = ~mask)

print('DQN: {}\nAC: {}\nDDQN: {}\nRain: {}'.format(masked_results_dqn.mean(),masked_results_ac.mean(),masked_results_ddqn.mean(),
masked_results_rain.mean()))


plt.show()

# # Actor Critic 2-Agent Full
# directory = "ActorCritic_2Agent_Full/"
# Agent = AC_2agent(args,'full')
# # Agent.load(directory,'')
# Agent.run_training(directory,100)
# # Agent.evaluate(directory,True)
# # Agent.render_run(10)

# Actor Critic 2-Agent Partial
# directory = "ActorCritic_2Agent_Partial/"
# Agent = AC_2agent(args,'partial')
# # Agent.load(directory)
# Agent.run_training(directory,1000)
# Agent.evaluate(directory,True)
