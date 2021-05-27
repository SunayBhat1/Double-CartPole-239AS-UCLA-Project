from agent import Agent
import numpy as np
from carts_poles import CartsPolesEnv
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
        torch.save(self.actor.state_dict(), dirname + 'actor.pkl')
        torch.save(self.critic.state_dict(), dirname + 'critic.pkl')
        torch.save(self.rewards_history, dirname + 'reward_history.pkl')
        torch.save(self.actor.state_dict(), dirname + 'Archive/actor_' + time.strftime("%Y%m%d-%H%M%S") + '.pkl')
        torch.save(self.critic.state_dict(), dirname + 'Archive/critic_' + time.strftime("%Y%m%d-%H%M%S") + '.pkl')
        torch.save(self.rewards_history, dirname + 'Archive/reward_history_' + time.strftime("%Y%m%d-%H%M%S") + '.pkl')

        print('Model saved to {}'.format(dirname))
    
    def load(self, dirname: str) -> None:
        a_model = torch.load(dirname + 'actor.pkl')
        c_model = torch.load(dirname + 'critic.pkl')
        self.actor.load_state_dict(a_model)
        self.critic.load_state_dict(c_model)
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

        env = CartsPolesEnv()

        optimizerA = optim.Adam(self.actor.parameters(),lr=self.alpha)
        optimizerC = optim.Adam(self.critic.parameters(),lr=self.alpha)
        rewards = self.rewards_history
        times = []

        for episode in tqdm(range(self.n),ncols=100):

            Angle = (np.random.rand()*2*self.rand_angle)-self.rand_angle
            state = env.reset(Angle)

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
                    tqdm.write('Episode: {} Maxed out Time!'.format(episode))
                    self.rewards_history = rewards
                    self.save(dirname)
                    break

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
            if (episode % print_log == 0): tqdm.write('Episode: {}, Seconds: {:.4f}, Start Angle: {:.4f}'.format(episode, info['time'], Angle))

        self.rewards_history = rewards
        self.save(dirname)
        self.plot_training(rewards,self.mean_window,self.method,dirname)
        env.close()
        print('Done Training!'.format())
        # self.evaluate(True)

        # end def run_training

    def evaluate(self, dirname: str, plot: bool) -> None:

        env = CartsPolesEnv()

        tot_rewards = np.zeros(np.shape(self.test_angles)[0])

        for i,iAngle in enumerate(tqdm(self.test_angles)):
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
            