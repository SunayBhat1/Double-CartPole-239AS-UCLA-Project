from agent import Agent
import gym
import numpy as np
from carts_poles import CartsPolesEnv
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden1_dim, hidden2_dim):
        super(Actor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear1 = nn.Linear(self.input_dim, hidden1_dim)
        self.linear2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.linear3 = nn.Linear(hidden2_dim, self.output_dim)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution

class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden1_dim, hidden2_dim):
        super(Critic, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear1 = nn.Linear(self.input_dim, hidden1_dim)
        self.linear2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.linear3 = nn.Linear(hidden2_dim, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value

class AC_agent(Agent):
    # method = "AC"
    # env=CartsPolesEnv()

    def __init__(self,args):
        self.method = "AC"
        self.n = args['n_episode']
        self.alpha = args['alpha']
        self.gamma = args['gamma']
        self.rand_angle = args['rand_angle']
        self.nn_hidden1_dim = args['ac_hidden1_dim']
        self.nn_hidden2_dim = args['ac_hidden2_dim']
        self.mean_window = args['mean_window']
        self.horizon=args['horizon']
        self.test_angles = args['test_angles']

        env=CartsPolesEnv()
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.actor = Actor(self.input_dim, self.output_dim,self.nn_hidden1_dim,self.nn_hidden2_dim)
        self.critic = Critic(self.input_dim, self.output_dim,self.nn_hidden1_dim,self.nn_hidden2_dim)

        self.rewards_history = []

    def save(self, dirname: str) -> None:
        torch.save(self.actor.state_dict(), dirname + 'actor.pt')
        torch.save(self.critic.state_dict(), dirname + 'critic.pt')
        torch.save(self.rewards_history, dirname + 'reward_history.pkl')
        torch.save(self.actor.state_dict(), dirname + 'Archive/actor_' + time.strftime("%Y%m%d-%H%M%S") + '.pt')
        torch.save(self.critic.state_dict(), dirname + 'Archive/critic_' + time.strftime("%Y%m%d-%H%M%S") + '.pt')
        torch.save(self.rewards_history, dirname + 'Archive/reward_history_' + time.strftime("%Y%m%d-%H%M%S") + '.pkl')

        print('Model saved to {}'.format(dirname))
    
    def load(self, dirname: str, file_ext: str) -> None:
        a_model = torch.load(dirname + 'actor.pt')
        c_model = torch.load(dirname + 'critic.pt')
        self.actor.load_state_dict(a_model)
        self.critic.load_state_dict(c_model)
        self.rewards_history = torch.load(dirname + 'reward_history.pkl')

        print('Model loaded from {}'.format(dirname))

    def plot_training(self,dirname):
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,4.5), dpi= 120, facecolor='w', edgecolor='k')
        fig.suptitle('Training Performance\n\n',fontweight='bold',fontsize = 14)

        ax1.plot(range(0,len(self.rewards_history)),self.rewards_history)
        ax1.set_title("Rewards vs Episode",fontweight='bold',fontsize = 11)
        ax1.set_xlabel('Episode',fontweight='bold',fontsize = 8)
        ax1.set_ylabel('Episode Rewards',fontweight='bold',fontsize = 8)
        ax1.grid()
        
        ax2.plot(range(self.mean_window-1,len(self.rewards_history)), np.convolve(self.rewards_history, np.ones(self.mean_window)/self.mean_window, mode='valid'),c='m')
        ax2.set_title("{} Avg Rewards vs Episode".format(self.mean_window),fontweight='bold',fontsize = 11)
        ax2.set_xlabel('Last Episode',fontweight='bold',fontsize = 8)
        ax2.set_ylabel('Mean 100 Rewards',fontweight='bold',fontsize = 8)
        ax2.grid()

        fig.savefig(dirname + 'Plots/Training_' + time.strftime("%Y%m%d-%H%M%S") + '.png')
        plt.pause(0.001)
    
    def compute_returns(self, next_value, rewards, masks):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def plot_compTime(self, comp_times,dirname):
        iter_times = np.diff(np.array(comp_times))
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,4.5), dpi= 120, facecolor='w', edgecolor='k')
        fig.suptitle('Computation Performance\n\n',fontweight='bold',fontsize = 14)

        ax1.plot(range(0,len(comp_times)),comp_times)
        ax1.set_title("Cumulative Time vs Episode",fontweight='bold',fontsize = 11)
        ax1.set_xlabel('Episode',fontweight='bold',fontsize = 8)
        ax1.set_ylabel('Cumulative Time (Seconds)',fontweight='bold',fontsize = 8)
        ax1.grid()

        ax2.plot(range(0,len(iter_times)),iter_times)        
        ax2.set_title("Iteration Time vs Episode",fontweight='bold',fontsize = 11)
        ax2.set_xlabel('Episode',fontweight='bold',fontsize = 8)
        ax2.set_ylabel('Time Per Iteration (Seconds)',fontweight='bold',fontsize = 8)  
        ax2.grid()

        fig.savefig(dirname + 'Plots/Computation_' + time.strftime("%Y%m%d-%H%M%S") + '.png')
        plt.pause(0.001)

    def run_training(self, dirname: str, print_log: int) -> None:

        env = CartsPolesEnv()

        optimizerA = optim.Adam(self.actor.parameters(),lr=self.alpha)
        optimizerC = optim.Adam(self.critic.parameters(),lr=self.alpha)
        rewards = self.rewards_history
        times = []
        comp_times = []

        start_time = time.time()
        for episode in tqdm(range(self.n),ncols=100):

            angle = (np.random.rand()*2*self.rand_angle)-self.rand_angle
            state = env.reset(angle)

            log_probs = []
            values = []
            reward_train = []
            masks = []
            entropy = 0

            ep_reward = 0

            done = False

            while not done:
                # env.render()
                state = torch.FloatTensor(state)
                dist, value = self.actor(state), self.critic(state)

                action = dist.sample()
                next_state, reward, done, info = env.step(action)

                log_prob = dist.log_prob(action).unsqueeze(0)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                reward_train.append(torch.tensor([reward], dtype=torch.float))
                masks.append(torch.tensor([1-done], dtype=torch.float))

                state = next_state

                ep_reward += reward

                if info['time'] > self.horizon:
                    tqdm.write('Episode: {} Maxed out Train Time (200s)!'.format(episode))
                    self.rewards_history = rewards
                    self.save(dirname)
                    break

            comp_times.append(time.time() - start_time)
            
            next_state = torch.FloatTensor(next_state)
            next_value = self.critic(next_state)
            returns = self.compute_returns(next_value, reward_train, masks)

            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns).detach()
            values = torch.cat(values)

            advantage = returns - values

            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()

            optimizerA.zero_grad()
            optimizerC.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            optimizerA.step()
            optimizerC.step()

            rewards.append(ep_reward)
            times.append(info['time'])
            
            if np.mean(times[-100:])>195:
                tqdm.write('Episode: {} 5 Ep Avg Greater than 400! Breaking Training'.format(episode))
                break
            if (episode % print_log == 0): tqdm.write('Episode: {}, Seconds: {:.4f}, Start Angle: {:.4f}'.format(episode, info['time'], angle))

        self.rewards_history = rewards
        self.save(dirname)
        self.plot_training(dirname)
        self.plot_compTime(comp_times,dirname)
        env.close()
        print('Done Training {} episodes!'.format(self.n))
        # self.evaluate(True)

        # end def run_training

    def evaluate(self, dirname: str, plot: bool) -> float:

        env = CartsPolesEnv()

        tot_time = np.zeros(np.shape(self.test_angles)[0])

        for i,iAngle in enumerate(tqdm(self.test_angles,ncols=100)):
            s = env.reset(iAngle)
            done = False
            ep_rewards = 0

            duration = 0

            while (duration <= self.horizon):
                state = torch.FloatTensor(s)
                dist = self.actor(state)
                a = dist.sample()
                s, r, done, info = env.step(a)

                duration = info['time']

                ep_rewards += r

                if done: break

            tot_time[i] = duration
            env.close()

        if plot: 
            mask = abs(self.test_angles * 180/np.pi) < 12
            masked_results = np.ma.array(tot_time,mask = ~mask)

            fig, ax0 = plt.subplots(figsize=(6,4), dpi= 130, facecolor='w', edgecolor='k')
            ax0.plot(self.test_angles * 180/np.pi,tot_time,c='g')
            ax0.set_title("Start Angle vs Episode Length\nMean (-12 to 12 Degrees): {:.2f}".format(masked_results.mean()),fontweight='bold',fontsize = 14)
            ax0.set_ylabel("Episode Length (Seconds)",fontweight='bold',fontsize = 12)
            ax0.set_xlabel("Start Angle (Degrees)",fontweight='bold',fontsize = 12)
            ax0.grid()
            fig.savefig(dirname + 'Plots/Results_' + time.strftime("%Y%m%d-%H%M%S") + '.png')
            # plt.show()

        return tot_time

    def render_run(self,dirname,save_video = False,speed=1,iters = 1,fps=20) -> None:

        env = CartsPolesEnv()

        for iEp in range(iters):
            if save_video: video_out = cv2.VideoWriter(dirname + 'Videos/Run_{}_{}xSpeed.mp4'.format(iEp,speed), cv2.VideoWriter_fourcc(*'mp4v'), fps*speed, (2000,1400))
            angle = (np.random.rand()*2*self.rand_angle)-self.rand_angle
            s = env.reset(angle)

            done = False
            iFrame = 0
            while not done:
                if save_video and (iFrame % (100/fps) == 0): 
                    img = env.render('rgb_array')
                    video_out.write(img)
                else: env.render()
                
                state = torch.FloatTensor(s)
                dist = self.actor(state)
                a = dist.sample()
                s, _, done, info = env.step(a)
                iFrame += 1

                if(info["time"]>=200): done = True
                if(iFrame % 2500 == 0): print('Time Elapsed = {:.4f} Seconds'.format(info["time"]))

            if save_video: video_out.release()
   
        print('Final Start Angle {:.4f}, Final Run Time: {:.2f}'.format(angle,info['time']))
        if save_video: print('Video saved to "' + dirname + 'Videos/"...')
        env.close()


            