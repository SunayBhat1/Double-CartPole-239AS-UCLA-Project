from numpy import median
import torch
from torch import nn
from agent import Agent, Memory
from carts_poles import CartsPolesEnv
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random
import tqdm as tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

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
    env=CartsPolesEnv()
    model="DDQN"
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
        self.Q_1 = QNetwork(action_dim=DDQN_agent.env.action_space.n, state_dim=DDQN_agent.env.observation_space.shape[0],
                                        hidden_dim=self.hidden_dim)
        self.Q_2 = QNetwork(action_dim=DDQN_agent.env.action_space.n, state_dim=DDQN_agent.env.observation_space.shape[0],
                                        hidden_dim=self.hidden_dim)
        for param in self.Q_2.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.Adam(self.Q_1.parameters(), lr=1e-3)
        self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=1)

    def load(self, dirname: str) -> None:
        model=torch.load(dirname)
        self.Q_1.load_state_dict(model)
        self.update_parameters(self.Q_1, self.Q_2)
        for param in self.Q_2.parameters():
            param.requires_grad = False

    def save(self, dirname: str) -> None:
        torch.save(self.Q_1.state_dict(), dirname)
    
    def evaluate(self, dirname: str, plot: bool) -> None:
        env = CartsPolesEnv()

        tot_rewards = np.zeros(np.shape(self.test_angles)[0])

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

            tot_rewards[i] = duration
            env.close()
        if plot: 
            fig, ax0 = plt.subplots(figsize=(6,3.5), dpi= 130, facecolor='w', edgecolor='k')
            ax0.plot(self.test_angles,tot_rewards,c='g')
            ax0.set_title("Start Angle vs Episode Length",fontweight='bold',fontsize = 15)
            ax0.set_ylabel("Episode Length (Seconds)",fontweight='bold',fontsize = 12)
            ax0.set_xlabel("Start Angle (Radians)",fontweight='bold',fontsize = 12)
            ax0.grid()
            fig.savefig('Plots/' + '_Results_' + time.strftime("%Y%m%d-%H%M%S") + '.png')
            plt.show()
    
    def run_training(self, dirname: str, print_log: int) -> None:
        memory = Memory(self.capacity)
        performance = []
        stop = 0
        biggest = 0
        measure_step=100
        eps=1
        for episode in range(self.n_episode):
            # display the performance
            stop+=1
            if (episode+1) % measure_step == 0:
                performance.append([episode, self.evaluate_MC()[1]])
                if biggest < performance[-1][1]:
                    biggest = performance[-1][1]
                print("Episode: ", episode)
                print("rewards: ", performance[-1][1])
                print("lr: ", self.scheduler.get_lr()[0])
                print("eps: ", eps)
                if performance[-1][1]>=self.horizon-50:
                    print("Ended early!!!!")
                    break
            randAngle =  (np.random.rand()*2*self.rand_angle)-self.rand_angle
            state = DDQN_agent.env.reset(randAngle)
            memory.state.append(state)

            done = False
            i = 0
            while not done:
                
                action = self.select_action(state, eps)
                state, reward, done, _ = self.env.step(action)
                i += reward
                if i > self.horizon:
                    done = True

                # save state, action, reward sequence
                memory.update(state, action, reward, done)

            if memory.length()>self.batch_size and episode % 50 == 0:
                for _ in range(50):
                    self.train(memory)

                # transfer new parameter from Q_1 to Q_2
                self.update_parameters(self.Q_1, self.Q_2)

            # update learning rate and eps
            self.scheduler.step()
            eps_decay=(1-self.min_eps)/self.max_episode
            eps = max(eps*eps_decay, self.min_eps)
    
    def plot_training(self, rewards, times) -> None:
        return super().plot_training(rewards, times)
    
    # Helper functions not required by the agent class
    def update_parameters(self, current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())
    
    def evaluate_MC(self):
        self.Q_1.eval()
        perform = 0
        repeats=100
        for _ in range(repeats):
            tot_reward=0
            randAngle =  (np.random.rand()*2*self.rand_angle)-self.rand_angle
            state = DDQN_agent.env.reset(randAngle)
            done = False
            while not done:
                state = torch.Tensor(state)
                with torch.no_grad():
                    values = self.Q_1(state)
                action = np.argmax(values.numpy())
                state, reward, done, _ = DDQN_agent.env.step(action)
                perform += reward
                tot_reward+=reward
                if tot_reward>self.horizon:
                    done=True
                
        self.Q_1.train()
        if perform/repeats> .95:
            return True, perform/repeats
        return False, perform/repeats

    def select_action(self, state, eps):
        state = torch.Tensor(state)
        with torch.no_grad():
            values = self.Q_2(state)

        # select a random action wih probability eps
        if random.random() <= eps:
            action = np.random.randint(0, DDQN_agent.env.action_space.n)
        else:
            action = np.argmax(values.numpy())

        return action
    
  
    def train(self, memory):

        states, actions, next_states, rewards, is_done = memory.sample(self.batch_size)

        q_values = self.Q_1(states)

        next_q_values = self.Q_1(next_states)
        next_q_state_values = self.Q_2(next_states)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + self.gamma * next_q_value * (1 - is_done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
