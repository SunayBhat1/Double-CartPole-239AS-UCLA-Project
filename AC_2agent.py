from agent import Agent
import numpy as np
from carts_poles_2agent import CartsPoles2Env
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

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

class AC_2agent(Agent):
    # method = "AC"
    # env=CartsPolesEnv()

    def __init__(self,args,infostate='full'):
        self.method = "2Agent_AC"
        self.n = args['n_episode']
        self.alpha = args['alpha']
        self.gamma = args['gamma']
        self.rand_angle = args['rand_angle']
        self.nn_hidden1_dim = args['ac_hidden1_dim']
        self.nn_hidden2_dim = args['ac_hidden2_dim']
        self.mean_window = args['mean_window']
        self.horizon=args['horizon']
        self.test_angles = args['test_angles']
        self.infostate = infostate

        env = CartsPoles2Env(self.infostate)
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.actor1 = Actor(self.input_dim, self.output_dim,self.nn_hidden1_dim,self.nn_hidden2_dim)
        self.actor2 = Actor(self.input_dim, self.output_dim,self.nn_hidden1_dim,self.nn_hidden2_dim)
        self.critic1 = Critic(self.input_dim, self.output_dim,self.nn_hidden1_dim,self.nn_hidden2_dim)
        self.critic2 = Critic(self.input_dim, self.output_dim,self.nn_hidden1_dim,self.nn_hidden2_dim)

        self.rewards_history = []

    def save(self, dirname: str) -> None:
        torch.save(self.actor1.state_dict(), dirname + 'actor1.pkl')
        torch.save(self.actor2.state_dict(), dirname + 'actor2.pkl')
        torch.save(self.critic1.state_dict(), dirname + 'critic1.pkl')
        torch.save(self.critic2.state_dict(), dirname + 'critic2.pkl')
        torch.save(self.rewards_history, dirname + 'reward_history.pkl')
        torch.save(self.actor1.state_dict(), dirname + 'Archive/actor1_' + time.strftime("%Y%m%d-%H%M%S") + '.pkl')
        torch.save(self.actor2.state_dict(), dirname + 'Archive/actor2_' + time.strftime("%Y%m%d-%H%M%S") + '.pkl')
        torch.save(self.critic1.state_dict(), dirname + 'Archive/critic1_' + time.strftime("%Y%m%d-%H%M%S") + '.pkl')
        torch.save(self.critic2.state_dict(), dirname + 'Archive/critic2_' + time.strftime("%Y%m%d-%H%M%S") + '.pkl')
        torch.save(self.rewards_history, dirname + 'Archive/reward_history_' + time.strftime("%Y%m%d-%H%M%S") + '.pkl')

        print('Model saved to {}'.format(dirname))
    
    def load(self, dirname: str) -> None:
        a1_model = torch.load(dirname + 'actor1.pkl')
        a2_model = torch.load(dirname + 'actor2.pkl')
        c1_model = torch.load(dirname + 'critic1.pkl')
        c2_model = torch.load(dirname + 'critic2.pkl')
        self.actor1.load_state_dict(a1_model)
        self.actor2.load_state_dict(a2_model)
        self.critic1.load_state_dict(c1_model)
        self.critic2.load_state_dict(c2_model)
        self.rewards_history = torch.load(dirname + 'reward_history.pkl')

        print('Model loaded from {}'.format(dirname))

    def plot_training(self, rewards, mean_window,method,dirname) -> None:
        return super().plot_training(rewards, mean_window,method,dirname)

    def compute_returns(self, next_value, rewards, masks):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def run_training(self, dirname: str, print_log: int) -> None:

        env = CartsPoles2Env(self.infostate)

        optimizerA1 = optim.Adam(self.actor1.parameters(),lr=self.alpha)
        optimizerA2 = optim.Adam(self.actor2.parameters(),lr=self.alpha)
        optimizerC1 = optim.Adam(self.critic1.parameters(),lr=self.alpha)
        optimizerC2 = optim.Adam(self.critic2.parameters(),lr=self.alpha)
        rewards = self.rewards_history
        times = []

        for episode in tqdm(range(self.n),ncols=100):

            Angle = (np.random.rand()*2*self.rand_angle)-self.rand_angle
            
            if self.infostate == 'full':
                state = env.reset(Angle)
            else:
                state1,state2 = env.reset(Angle)

            log_probs1 = []
            log_probs2 = []
            values1 = []
            values2 = []
            reward_train = []
            masks = []
            entropy1 = 0
            entropy2 = 0

            ep_reward = 0

            done = False

            while not done:
                # env.render()
                if self.infostate == 'full':
                    state = torch.FloatTensor(state)
                    dist1, dist2, value1, value2 = self.actor1(state), self.actor2(state), self.critic1(state), self.critic2(state)
                else:
                    state1 = torch.FloatTensor(state1)
                    state2 = torch.FloatTensor(state2)
                    dist1, dist2, value1, value2 = self.actor1(state1), self.actor2(state2), self.critic1(state1), self.critic2(state2)

                action1 = dist1.sample()
                action2 = dist2.sample()

                if self.infostate == 'full':
                    next_state, reward, done, info = env.step(action1,action2)
                else:
                    next_state1, next_state2, reward, done, info = env.step(action1,action2)

                log_prob1 = dist1.log_prob(action1).unsqueeze(0)
                entropy1 += dist1.entropy().mean()

                log_prob2 = dist2.log_prob(action2).unsqueeze(0)
                entropy2 += dist2.entropy().mean()

                log_probs1.append(log_prob1)
                log_probs2.append(log_prob2)
                values1.append(value1)
                values2.append(value2)

                reward_train.append(torch.tensor([reward], dtype=torch.float))
                masks.append(torch.tensor([1-done], dtype=torch.float))

                if self.infostate == 'full':
                    state = next_state
                else:
                    state1 = next_state1
                    state2 = next_state2

                ep_reward += reward

                if info['time'] > self.horizon:
                    tqdm.write('Episode: {} Maxed out Time!'.format(episode))
                    self.rewards_history = rewards
                    self.save(dirname)
                    break

            if self.infostate == 'full':
                next_state = torch.FloatTensor(next_state)
                next_value1 = self.critic1(next_state)
                next_value2 = self.critic2(next_state)

            else:
                next_state1 = torch.FloatTensor(next_state1)
                next_state2 = torch.FloatTensor(next_state2)
                next_value1 = self.critic1(next_state1)
                next_value2 = self.critic2(next_state2)

            returns1 = self.compute_returns(next_value1, reward_train, masks)
            returns2 = self.compute_returns(next_value2, reward_train, masks)

            log_probs1 = torch.cat(log_probs1)
            log_probs2 = torch.cat(log_probs2)
            returns1 = torch.cat(returns1).detach()
            returns2 = torch.cat(returns2).detach()
            values1 = torch.cat(values1)
            values2 = torch.cat(values2)

            advantage1 = returns1 - values1
            advantage2 = returns2 - values2

            actor1_loss = -(log_probs1 * advantage1.detach()).mean()
            actor2_loss = -(log_probs2 * advantage2.detach()).mean()
            critic1_loss = advantage1.pow(2).mean()
            critic2_loss = advantage2.pow(2).mean()

            optimizerA1.zero_grad()
            optimizerA2.zero_grad()
            optimizerC1.zero_grad()
            optimizerC2.zero_grad()
            actor1_loss.backward()
            actor2_loss.backward()
            critic1_loss.backward()
            critic2_loss.backward()
            optimizerA1.step()
            optimizerA2.step()
            optimizerC1.step()
            optimizerC2.step()

            rewards.append(ep_reward)
            times.append(info['time'])
            if (episode % print_log == 0): tqdm.write('Episode: {}, Seconds: {:.4f}, Start Angle: {:.4f}'.format(episode, info['time'], Angle))

        self.rewards_history = rewards
        self.save(dirname)
        self.plot_training(rewards,self.mean_window,self.method,dirname)
        env.close()
        print('Done Training!'.format())

        # end def run_training

    def evaluate(self, dirname: str, plot: bool) -> None:

        env = CartsPoles2Env(self.infostate)

        tot_rewards = np.zeros(np.shape(self.test_angles)[0])

        for i,iAngle in enumerate(tqdm(self.test_angles)):
            if self.infostate == 'full':
                s = env.reset(iAngle)
            else:
                s1,s2 = env.reset(iAngle)

            done = False
            ep_rewards = 0

            duration = 0

            while (duration <= self.horizon):
                if self.infostate == 'full':
                    state = torch.FloatTensor(s)
                    dist1 = self.actor1(state)
                    dist2 = self.actor2(state)
                else:
                    state1 = torch.FloatTensor(s1)
                    state2 = torch.FloatTensor(s2)
                    dist1 = self.actor1(state1)
                    dist2 = self.actor2(state2)


                a1 = dist1.sample()
                a2 = dist2.sample()

                if self.infostate == 'full': s, r, done, info = env.step(a1,a2)
                else: s1, s2, r, done, info = env.step(a1,a2)

                duration = info['time']

                ep_rewards += r

                if done: break 

            tot_rewards[i] = duration
            env.close()

        if plot: 
            fig, ax0 = plt.subplots(figsize=(6,4), dpi= 130, facecolor='w', edgecolor='k')
            ax0.plot(self.test_angles,tot_rewards,c='g')
            ax0.set_title("Start Angle vs Episode Length",fontweight='bold',fontsize = 15)
            ax0.set_ylabel("Episode Length (Seconds)",fontweight='bold',fontsize = 12)
            ax0.set_xlabel("Start Angle (Radians)",fontweight='bold',fontsize = 12)
            ax0.grid()
            fig.savefig(dirname + 'Plots/' + self.method + '_Results_' + time.strftime("%Y%m%d-%H%M%S") + '.png')
            plt.show()
            