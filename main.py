from DQN_agent import DQN_agent
from DDQN_agent import DDQN_agent
from Rainbow_agent import Rainbow_agent
from AC_agent import AC_agent
from AC_2agent import AC_2agent
import numpy as np
import matplotlib.pyplot as plt
from urllib import request
notify = True

args = {
        "n_episode" : 15000, # All
        # Default 0.99
        "gamma" : 0.9, # All
        # Default 0.2
        "rand_angle" : 0.2, # All
        # Default: 100
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
        "max_episode" : 50, # DQN, DDQN
        # Default: 0.01
        "min_eps" : 0.01, # DQN, DDQN
        # Default: 128
        'ac_hidden1_dim' : 128, # AC
        # Default: 256
        'ac_hidden2_dim': 256, # AC
        # Default:  0.0001
        'alpha': 0.0001, # AC

        'n_step_return': 5,
        'replay_start_size': 42,

        'load_path': '/Rainbow/model.pt',

        'steps': 100
    }



# # Rainbow Agent
# Agent=Rainbow_agent(args)


# # DQN Agent
directory = "DQN/"
Agent=DQN_agent(args)
# Agent.run_training("",100)
Agent.load('DQN/DQN_5_20_0010.pt')
vec_dqn = Agent.evaluate(True)

args['hidden_dim'] = 64
# DDQN Agent
directory = "DDQN/"
Agent=DDQN_agent(args)
# Agent.run_training("",100)
Agent.load(directory + 'DDQN_Q1.pt',file_ext='')
vec_ddqn = Agent.evaluate(directory,True)
# Agent.render_run(directory,save_video=False)

# Actor Critic Agent
directory = "ActorCritic/"
Agent=AC_agent(args)
Agent.load(directory,'')
# Agent.run_training(directory,500)
vec_ac = Agent.evaluate(directory,True)
# Agent.render_run(directory,True,1)

fig7, ax9 = plt.subplots(figsize=(6,3.5), dpi= 130, facecolor='w', edgecolor='k')
ax9.plot(args['test_angles']*180/np.pi,vec_ddqn,c='g')
ax9.plot(args['test_angles']*180/np.pi,vec_ac,c='b')
ax9.plot(args['test_angles']*180/np.pi,vec_dqn,c='r')
ax9.set_title('Start Angle vs Episode Length',fontweight='bold',fontsize = 14)
ax9.set_ylabel("Episode Length (Seconds)",fontweight='bold',fontsize = 12)
ax9.set_xlabel("Start A ngle (Degrees)",fontweight='bold',fontsize = 12)
ax9.legend(['DDQN','AC','DQN'])
ax9.grid()
fig7.savefig('Results_All.png')
plt.show()


# Actor Critic 2-Agent Full
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


if notify:
    key = "Lmdwc3Ei4h0vfiIwLA4K0"
    message = 'Python_Script_Is_Done'
    request.urlopen("https://maker.ifttt.com/trigger/notify/with/key/%s?value1=%s" % (key,message))