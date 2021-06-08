import numpy as np
import torch
import torch.nn as nn
import pfrl
import gym
import pfrl.nn as pnn
import time
import matplotlib.pyplot as plt
from agent import Agent
from carts_poles import CartsPolesEnv
from tqdm import tqdm
from pfrl import agents, explorers
from pfrl import replay_buffers, utils

class MultiBinaryAsDiscreteAction(gym.ActionWrapper):
    """Transforms MultiBinary action space to Discrete.
    If the action space of a given env is `gym.spaces.MultiBinary(n)`, then
    the action space of the wrapped env will be `gym.spaces.Discrete(2**n)`,
    which covers all the combinations of the original action space.
    Args:
        env (gym.Env): Gym env whose action space is `gym.spaces.MultiBinary`.
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        self.orig_action_space = env.action_space
        self.action_space = gym.spaces.Discrete(2 ** env.action_space.n)

    def action(self, action):
        return [(action >> i) % 2 for i in range(self.orig_action_space.n)]


class DistributionalDuelingHead(nn.Module):
    """Head module for defining a distributional dueling network.
    This module expects a (batch_size, in_size)-shaped `torch.Tensor` as input
    and returns `pfrl.action_value.DistributionalDiscreteActionValue`.
    Args:
        in_size (int): Input size.
        n_actions (int): Number of actions.
        n_atoms (int): Number of atoms.
        v_min (float): Minimum value represented by atoms.
        v_max (float): Maximum value represented by atoms.
    """

    def __init__(self, in_size, n_actions, n_atoms, v_min, v_max):
        super().__init__()
        assert in_size % 2 == 0
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.register_buffer(
            "z_values", torch.linspace(v_min, v_max, n_atoms, dtype=torch.float)
        )
        self.a_stream = nn.Linear(in_size // 2, n_actions * n_atoms)
        self.v_stream = nn.Linear(in_size // 2, n_atoms)

    def forward(self, h):
        h_a, h_v = torch.chunk(h, 2, dim=1)
        a_logits = self.a_stream(h_a).reshape((-1, self.n_actions, self.n_atoms))
        a_logits = a_logits - a_logits.mean(dim=1, keepdim=True)
        v_logits = self.v_stream(h_v).reshape((-1, 1, self.n_atoms))
        probs = nn.functional.softmax(a_logits + v_logits, dim=2)
        return pfrl.action_value.DistributionalDiscreteActionValue(probs, self.z_values)



class Rainbow_agent(Agent):
    def __init__(self, args):
        print(args)
        self.env = CartsPolesEnv()
        self.train_seed = args['seed']
        self.n_step_return = args['n_step_return']
        self.gpu = args['gpu']
        self.gamma = args['gamma']
        self.replay_start_size = args['replay_start_size']
        self.load_path = args['load_path']
        self.n_episodes = args['n_episode']
        self.max_episode = args['max_episode']
        self.rand_angle = args['rand_angle']
        #self.save_path = args['save_path']
        self.test_angles = args['test_angles']
        self.steps = args['steps']

        obs_size = self.env.observation_space.low.size
        n_actions = self.env.action_space.n

        n_atoms = 51
        v_max = 1
        v_min = -v_max
        hidden_size = 512
        
        self.q_func = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            DistributionalDuelingHead(hidden_size, n_actions, n_atoms, v_min, v_max),
        )

        pnn.to_factorized_noisy(self.q_func, sigma_scale=0.1)
        self.explorer = explorers.Greedy()

        self.opt = torch.optim.Adam(self.q_func.parameters(), 1e-4, eps=1.5e-4)
        
        update_interval = 1
        betasteps = self.steps / update_interval
        
        self.rbuf = replay_buffers.PrioritizedReplayBuffer(
            10 ** 6,
            alpha=0.5,
            beta0=0.4,
            betasteps=betasteps,
            num_steps=self.n_step_return,
            normalize_by_max="memory",
        )

        self.agent = agents.CategoricalDoubleDQN(
            self.q_func,
            self.opt,
            self.rbuf,
            gpu=self.gpu,
            gamma=self.gamma,
            explorer=self.explorer,
            minibatch_size=32,
            replay_start_size=self.replay_start_size,
            target_update_interval=2000,
            update_interval=update_interval,
            batch_accumulator="mean",
            phi=self.phi,
            max_grad_norm=10,
        )

        
        if self.load_path != None:
            print("loading")
            self.load(self.load_path)
        
        
        utils.set_random_seed(args['seed'])


    def _check_env(self, env):
        if isinstance(env.action_space, gym.spaces.MultiBinary):
            env = MultiBinaryAsDiscreteAction(env)

        return

    def evaluate(self, dirname: str, plot: bool) -> None:

        tot_rewards = np.zeros(np.shape(self.test_angles)[0])
        print(self.agent) 
        with self.agent.eval_mode():
            for i, iAngle in enumerate(tqdm(self.test_angles)):
                obs = self.env.reset(iAngle)
                R = 0  # return (sum of rewards)
                t = 0  # time step
                while True:
                    # Uncomment to watch the behavior in a GUI window
                    # env.render()
                    action = self.agent.act(obs)
                    obs, reward, done, info = self.env.step(action)
                    R += reward
                    t = info['time']
                    reset = (t >= self.max_episode)
                    self.agent.observe(obs, reward, done, reset)
    
                    if done or reset:
                        break
                
                tot_rewards[i] = R

                print('[Evaluate] episode:', i, 'R:', R)
                
            np.save(dirname + 'evaluate.npy', tot_rewards) 
            if plot: 
                fig, ax0 = plt.subplots(figsize=(6,3.5), dpi= 130, facecolor='w', edgecolor='k')
                ax0.plot(self.test_angles, tot_rewards, c='g')
                ax0.set_title("Start Angle vs Episode Length",fontweight='bold',fontsize = 15)
                ax0.set_ylabel("Episode Length (Seconds)",fontweight='bold',fontsize = 12)
                ax0.set_xlabel("Start Angle (Radians)",fontweight='bold',fontsize = 12)
                ax0.grid()
                fig.savefig(dirname + '_Results_' + time.strftime("%Y%m%d-%H%M%S") + '.png')
                plt.show()
 
    def evaluate_MC(self) -> bool:
        return_list = []

        with self.agent.eval_mode():
            for i in range(100):
                randAngle =  (np.random.rand()*2*self.rand_angle)-self.rand_angle
                obs = self.env.reset(randAngle)
                R = 0  # return (sum of rewards)
                t = 0  # time step
                while True:
                    # Uncomment to watch the behavior in a GUI window
                    # env.render()
                    action = self.agent.act(obs)
                    obs, reward, done, info = self.env.step(action)
                    R += reward
                    t = info['time']
                    reset = (t >= self.max_episode)
                    self.agent.observe(obs, reward, done, reset)
    
                    if done or reset:
                        return_list.append(R)
                        break
    
                return_list.append(R)

        current_avg_return = np.array(return_list).mean()
        
        print('[Evaluate] Average Return: ', current_avg_return)
        
        if_better = (current_avg_return > self.best_performance)

        if if_better:
            self.best_performance = current_avg_return
            
        print('Evaluate MC Finished.')

        return if_better
            

    def run_training(self, dirname: str, print_log: int):
        best_performance = 0
        return_list = []

        for i in range(1, self.n_episodes + 1): 
            randAngle =  (np.random.rand()*2*self.rand_angle)-self.rand_angle
            obs = self.env.reset(randAngle)
            R = 0  # return (sum of rewards)
            t = 0  # time step
            while True:
                # Uncomment to watch the behavior in a GUI window
                # env.render()
                action = self.agent.act(obs)
                obs, reward, done, info = self.env.step(action)
                R += reward
                t = info['time']
                reset = (t >= self.max_episode)
                self.agent.observe(obs, reward, done, reset)

                if done or reset:
                    return_list.append(R)
                    break
            

            print('episode:', i, 'R:', R)
            #if i % 100 == 0 and print_log:
            #    print('episode:', i, 'R:', R)

            if i % 100 == 0:
            #    if_save = self.evaluate_MC()
            #    if if_save:
                
                #self.save(dirname)
                return_save = np.array(return_list)
                if return_save[-100:].mean()>best_performance:
                    best_performance = return_save[-100:].mean()
                    self.save(dirname)
                
                np.save(dirname + "training_log_ep_{}.npy".format(i), return_save)
            
            if i % 30 == 0 and print_log:
                print('statistics:', self.agent.get_statistics())
        
        print('Finished.')

    def save(self, dirname: str) -> None:
        self.agent.save(dirname)
        print("agent saved in path {}.".format(dirname))

    def load(self, dirname: str) -> None:
        self.agent.load(dirname)

    def plot_training(self, rewards, times) -> None:
        return super().plot_training(rewards, times)

    def phi(self, x):
        return np.asarray(x, dtype=np.float32)
