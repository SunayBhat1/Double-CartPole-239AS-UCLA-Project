from numpy import median
import torch
import time
from torch import nn
from agent import Agent
from carts_poles import CartsPolesEnv
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import torch.nn.functional as F
import collections
import time
import cv2

class Memory:
    def __init__(self, len):
        self.rewards = collections.deque(maxlen=len)
        self.state = collections.deque(maxlen=len)
        self.action = collections.deque(maxlen=len)
        self.is_done = collections.deque(maxlen=len)

    def update(self, state, action, reward, done):
        # if the episode is finished we do not save to new state. Otherwise we have more states per episode than rewards
        # and actions whcih leads to a mismatch when we sample from memory.
        if not done:
            self.state.append(state)
        self.action.append(action)
        self.rewards.append(reward)
        self.is_done.append(done)

    def sample(self, batch_size):
        """
        sample "batch_size" many (state, action, reward, next state, is_done) datapoints.
        """
        n = len(self.is_done)
        idx = random.sample(range(0, n-1), batch_size)

        return torch.Tensor(self.state)[idx], torch.LongTensor(self.action)[idx], \
               torch.Tensor(self.state)[1+np.array(idx)], torch.Tensor(self.rewards)[idx], \
               torch.Tensor(self.is_done)[idx]

    def reset(self):
        self.rewards.clear()
        self.state.clear()
        self.action.clear()
        self.is_done.clear()

class QNetwork(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, inp):

        x1 = F.leaky_relu(self.fc_1(inp))
        x1 = F.leaky_relu(self.fc_2(x1))
        x1 = self.fc_3(x1)

        return x1

class DDQN_agent(Agent):
    
    model="DDQN"
    def __init__(self, args):
        self.env=CartsPolesEnv()
        self.n_episode=args["n_episode"]
        self.batch_size=args["batch_size"]
        self.rand_angle = args['rand_angle']
        self.hidden_dim = args['hidden_dim']
        self.max_episode = args["max_episode"]
        self.min_eps = args["min_eps"]
        self.capacity = args["capacity"]
        self.mean_window = args['mean_window']
        self.horizon=args['horizon']
        self.test_angles = args['test_angles']
        self.gamma=args["gamma"]
        self.Q_1 = QNetwork(action_dim=self.env.action_space.n, state_dim=self.env.observation_space.shape[0],
                                        hidden_dim=self.hidden_dim)
        self.Q_2 = QNetwork(action_dim=self.env.action_space.n, state_dim=self.env.observation_space.shape[0],
                                        hidden_dim=self.hidden_dim)

    def load(self, dirname: str, file_ext:str="DDQN_Q1_working.pt") -> None:
        model=torch.load(dirname+file_ext)
        self.Q_1.load_state_dict(model)
        self.update_parameters()
        for param in self.Q_2.parameters():
            param.requires_grad = False

    def save(self, dirname: str) -> None:
        torch.save(self.Q_1.state_dict(), dirname+"DDQN_Q1.pt")
        torch.save(self.Q_1.state_dict(), dirname + 'Archive/DDQN_Q1' + time.strftime("%Y%m%d-%H%M%S") + '.pt')
    
    def evaluate(self, dirname: str, plot: bool=True) -> float:

        tot_time = np.zeros(np.shape(self.test_angles)[0])

        for i,iAngle in tqdm(enumerate(self.test_angles),ncols=100):
            s = self.env.reset(iAngle)
            done = False
            ep_rewards = 0

            duration = 0

            while (duration <= self.horizon):
                a=self.select_action(s,0)
                s2, r, done, info = self.env.step(a)

                duration = info['time']

                ep_rewards += r

                if done:
                    break 
                s = s2

            tot_time[i] = duration
        if plot:
            mask = abs(self.test_angles * 180/np.pi) < 12
            masked_results = np.ma.array(tot_time,mask = ~mask)

            fig, ax0 = plt.subplots(figsize=(6,3.5), dpi= 130, facecolor='w', edgecolor='k')
            ax0.plot(self.test_angles*180/np.pi,tot_time,c='g')
            ax0.set_title("Start Angle vs Episode Length\nMean (-12 to 12 Degrees): {:.2f}".format(masked_results.mean()),fontweight='bold',fontsize = 14)
            ax0.set_ylabel("Episode Length (Seconds)",fontweight='bold',fontsize = 12)
            ax0.set_xlabel("Start Angle (Degrees)",fontweight='bold',fontsize = 12)
            ax0.grid()
            fig.savefig(dirname+'Plots/Results' + time.strftime("%Y%m%d-%H%M%S") + '.png')
            # plt.show()
        return tot_time
    
    def run_training(self, dirname: str, print_log: int) -> None:
        self.update_parameters()

        for param in self.Q_2.parameters():
            param.requires_grad = False
        
        optimizer = torch.optim.Adam(self.Q_1.parameters(), lr=1e-3)
        scheduler = StepLR(optimizer, step_size=1000, gamma=1)

        memory = Memory(self.capacity)
        performance = []
        history=[]
        stop = 0
        biggest = 0
        measure_step=100
        eps=1
        for episode in tqdm(range(self.n_episode)):
            stop+=1
            if (episode+1) % measure_step == 0:
                eval_result=self.evaluate(dirname, False)
                performance.append([episode,eval_result])
                if eval_result>=biggest:
                    self.save(dirname)
                    biggest = eval_result
            if (episode+1) % print_log == 0:
                tqdm.write('Episode: {}, Seconds: {:.4f}, Learning Rate: {:.4f}, epsilon: {:.4f}'.format(episode,  performance[-1][1], scheduler.get_lr()[0],eps))


            randAngle =  (np.random.rand()*2*self.rand_angle)-self.rand_angle
            state = self.env.reset(randAngle)
            memory.state.append(state)

            done = False
            i = 0
            while not done:
                action = self.select_action(state, eps)
                state, reward, done, info= self.env.step(action)
                i = info['time']
                if i > self.horizon:
                    done = True

                # save state, action, reward sequence
                memory.update(state, action, reward, done)
            history.append([info['time']])
            if episode>=20 and episode % 10== 0:
                for _ in range(50):
                    self.train(memory,optimizer)
                # transfer new parameter from Q_1 to Q_2
                self.update_parameters()

            # update learning rate and eps
            warnings.filterwarnings("ignore")
            scheduler.step()
            eps_decay=(1-self.min_eps)/self.max_episode
            eps = max(eps*eps_decay, self.min_eps)
        # Plot training History
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,4.5), dpi= 120, facecolor='w', edgecolor='k')
        fig.suptitle('Training Performance\n\n',fontweight='bold',fontsize = 14)

        ax1.plot(range(0,len(history)),history)
        ax1.set_title("Rewards vs Episode",fontweight='bold',fontsize = 11)
        ax1.set_xlabel('Episode',fontweight='bold',fontsize = 8)
        ax1.set_ylabel('Episode Rewards',fontweight='bold',fontsize = 8)
        ax1.grid()
        
        ax2.plot([x[0] for x in performance],[x[1] for x in performance])
        ax2.set_title("{} Performance vs Episode".format(self.mean_window),fontweight='bold',fontsize = 11)
        ax2.set_xlabel('Episode',fontweight='bold',fontsize = 8)
        ax2.set_ylabel('Mean 100 Rewards',fontweight='bold',fontsize = 8)
        ax2.grid()

        fig.savefig(dirname + 'Plots/Training_' + time.strftime("%Y%m%d-%H%M%S") + '.png')
        plt.pause(0.001)
        
    
    # Helper functions not required by the agent class
    def update_parameters(self):
        self.Q_2.load_state_dict(self.Q_1.state_dict())
    

    def select_action(self, state, eps):
        state = torch.Tensor(state)
        with torch.no_grad():
            values = self.Q_2(state)

        # select a random action wih probability eps
        if random.random() <= eps:
            action = np.random.randint(0, self.env.action_space.n)
        else:
            action = np.argmax(values.numpy())

        return action
    
  
    def train(self, memory, optimizer):
        #batch_size, current, target, optim, memory, gamma
        #batch_size, Q_1, Q_2, optimizer, memory, gamma
        states, actions, next_states, rewards, is_done = memory.sample(self.batch_size)

        q_values = self.Q_1(states)

        next_q_values = self.Q_1(next_states)
        next_q_state_values = self.Q_2(next_states)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + self.gamma * next_q_value * (1 - is_done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def render_run(self, dirname = "DDQN/", save_video = False, speed=1,iters = 1,fps=20) -> None:

        env = CartsPolesEnv()

        for iEp in range(iters):
            if save_video: video_out = cv2.VideoWriter(dirname + 'Videos/Run_{}_{}xSpeed.mp4'.format(iEp,speed), cv2.VideoWriter_fourcc(*'mp4v'), fps*speed, (2000,1400))
            angle = (np.random.rand()*2*self.rand_angle)-self.rand_angle
            # angle = 0
            s = env.reset(angle)

            done = False
            iFrame = 1
            while not done:
                if save_video and (iFrame % (100/fps) == 0): 
                    img = env.render('rgb_array')
                    video_out.write(img)
                else: env.render()
                
                a=self.select_action(s,-1)
                s, _, done, info = env.step(a)
                iFrame +=1

                if(info["time"]>=200): done = True
                if(iFrame % 2500 == 0): print('Time Elapsed = {:.4f} Seconds'.format(info["time"]))
                    
            if save_video: video_out.release()
   
        print('Final Start Angle {:.4f}, Final Run Time: {:.2f}'.format(angle,info['time']))
        if save_video: print('Video saved to "' + dirname + 'Videos/"...')
        env.close()