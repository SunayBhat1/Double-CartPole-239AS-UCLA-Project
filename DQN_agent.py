from numpy.core.fromnumeric import repeat
from agent import Agent
from agent import Memory
from carts_poles import CartsPolesEnv
import torch
import torch.nn
import numpy as np
import random
import gym
from collections import namedtuple
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

class DQN(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        """DQN Network
        Args:
            input_dim (int): `state` dimension.
                `state` is 2-D tensor of shape (n, input_dim)
            output_dim (int): Number of actions.
                Q_value is 2-D tensor of shape (n, output_dim)
            hidden_dim (int): Hidden dimension in fc layer
        """
        super(DQN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.PReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.PReLU()
        )

        self.final = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a Q_value
        Args:
            x (torch.Tensor): `State` 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: Q_value, 2-D tensor of shape (n, output_dim)
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.final(x)

        return x

class DQN_agent(Agent):
    env=CartsPolesEnv()
    model="DQN"
    def __init__(self, args):
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
        
        self.input_dim = DQN_agent.env.observation_space.shape[0]
        self.output_dim = DQN_agent.env.action_space.n                  
        self.dqn = DQN(self.input_dim, self.output_dim, self.hidden_dim)
        
        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.dqn.parameters())

    def select_action(self, state: tuple, epsilon) -> int:
        if np.random.rand() < epsilon:
            return np.random.choice(self.output_dim)
        else:
            scores = self.get_Q(state)
            _, argmax = torch.max(scores.data, 1)
            return int(argmax.numpy())
            
    def evaluate_MC(self) -> bool:
        self.dqn.eval()
        repeats=100
        perform = 0
        for _ in range(repeats):
            tot_reward=0
            Angle = (np.random.rand()*2*self.rand_angle)-self.rand_angle
            state = DQN_agent.env.reset(Angle)
            done = False
            while not done:
                state = torch.Tensor(state)
                with torch.no_grad():
                    values = self.get_Q(state)
                action = np.argmax(values.cpu().numpy())
                state, reward, done, _ = DQN_agent.env.step(action)
                perform += reward
                tot_reward+=reward
                if tot_reward>self.horizon:
                    done=True
                
        self.dqn.train()
        if perform/repeats>=.99:
            return True, perform/repeats
        return False, perform/repeats

    def train(self,batch_size,replay_memory):
        states, actions, next_states, rewards, done = replay_memory.sample(batch_size)

        q_predict = self.dqn(states)
        q_target = q_predict.clone().data.numpy()

        q_target[np.arange(len(q_target)),actions] = rewards.numpy() + self.gamma * np.max(self.dqn(next_states).data.numpy(), axis=1) * (1-done.numpy())
        q_target = torch.autograd.Variable(torch.Tensor(q_target))

        self.dqn.train(mode=True)
        self.optim.zero_grad()
        loss = self.loss_fn(q_target, q_predict)
        loss.backward()
        self.optim.step()

    def run_training(self, dirname: str, print_log: int) -> None:
        rewards = np.zeros(self.n_episode)
        times = np.zeros(self.n_episode)
        replay_memory = Memory(self.capacity)
        avg = 0
        biggest_rs = 0
        for i in tqdm(range(self.n_episode)):
            #epsilon annealing
            slope = (self.min_eps - 1.0) / self.max_episode
            eps = max(slope * i + 1.0, self.min_eps)

            #r, history, info = play_episode(env, agent, replay_memory, eps,FLAGS['batch_size'])
            start = 0
            Angle = (np.random.rand()*2*self.rand_angle)-self.rand_angle
            s = DQN_agent.env.reset(Angle)
            done = False
            total_reward = 0
            while not done:
                a = self.select_action(s,eps)
                s2, r, done, info = DQN_agent.env.step(a)
                total_reward += r
                s = s2
                replay_memory.update(s, a, r, done)
                if info['time']>self.horizon:
                    done=True
            rewards[i] = total_reward
            times[i] = info['time']
            if replay_memory.length()>self.batch_size:
                    self.train(self.batch_size,replay_memory)
            if i % print_log==0:
                secounds=info['time']
                good, performance=self.evaluate_MC()
                tqdm.write('Episode: {}, Seconds: {:.4f}, Average Seconds: {:.4f}'.format(i,secounds, performance))
                if good:
                    torch.save(self.dqn.state_dict(),dirname+'dqn.pt')
                    torch.save(self.dqn.state_dict(), dirname + 'Archive/dqn' + time.strftime("%Y%m%d-%H%M%S") + '.pt')
        torch.save(self.dqn.state_dict(),dirname+'dqn_end.pkl')

    def save(self, dirname: str) -> None:
        torch.save(self.dqn.state_dict(), dirname)

    def render_run(self, dirname: str) -> None:
        print('do Nothing')
    
    def load(self, dirname: str) -> None:
        model=torch.load(dirname)
        self.dqn.load_state_dict(model)
        
    def plot_training(self, rewards, times) -> None:
        return super().plot_training(rewards, times)

    def evaluate(self,plot: bool) -> bool:
        env = CartsPolesEnv()

        tot_time = np.zeros(np.shape(self.test_angles)[0])

        for i,iAngle in enumerate(tqdm(self.test_angles)):
            s = env.reset(iAngle)
            done = False
            ep_rewards = 0
            prev = 0

            duration = 0

            while (duration <= self.horizon):
                a=self.select_action(s,0)
                s2, r, done, info = env.step(a)

                duration = info['time']

                ep_rewards += r

                if done:
                    break 
                s = s2

            tot_time[i] = duration
            env.close()
        if plot: 
            mask = abs(self.test_angles * 180/np.pi) < 12
            masked_results = np.ma.array(tot_time,mask = ~mask)

            fig, ax0 = plt.subplots(figsize=(6,3.5), dpi= 130, facecolor='w', edgecolor='k')
            ax0.plot(self.test_angles*180/np.pi,tot_time,c='g')
            ax0.set_title("Start Angle vs Episode Length\nMean (-12 to 12 Degrees): {:.2f}".format(masked_results.mean()),fontweight='bold',fontsize = 14)
            ax0.set_ylabel("Episode Length (Seconds)",fontweight='bold',fontsize = 12)
            ax0.set_xlabel("Start Angle (Degrees)",fontweight='bold',fontsize = 12)
            ax0.grid()
            fig.savefig('DQN/Plots/Results_' + time.strftime("%Y%m%d-%H%M%S") + '.png')
            # plt.show()
        return tot_time

    def get_Q(self, states: np.ndarray) -> torch.FloatTensor:
        """Returns `Q-value`
        Args:
            states (np.ndarray): 2-D Tensor of shape (n, input_dim)
        Returns:
            torch.FloatTensor: 2-D Tensor of shape (n, output_dim)
        """
        states = torch.autograd.Variable(torch.Tensor((states.reshape(-1, self.input_dim))))
        self.dqn.train(mode=False)
        states=self.dqn(states)
        self.dqn.train(mode=True)
        return states