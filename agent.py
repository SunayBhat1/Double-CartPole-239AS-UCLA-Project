from abc import ABCMeta, abstractmethod
from carts_poles import CartsPolesEnv # abstractproperty
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from collections import namedtuple
import collections
import torch

class Agent(object, metaclass=ABCMeta):

    @abstractmethod
    def save(self, dirname: str) -> None:
        """Save the state dictionary
        """
        raise NotImplementedError()

    @abstractmethod
    def load(self, dirname: str) -> None:
        """ Load in the network for this class
        """
        raise NotImplementedError()
    
    # @abstractmethod
    # def evaluate_MC(self) -> bool:
    #     """ Return the performance of the agent 
    #     that suffices to save the network
    #     """
    #     raise NotImplementedError()
    
    @abstractmethod
    def evaluate(self, dirname: str, plot: bool) -> None:
        """ Return the performance of the agent over all the angles in the range
        """
        raise NotImplementedError()

    @abstractmethod
    def run_training(self, dirname: str, print_log: int) -> None:
        """ Main loop, returns trained agent
            save agents that pass evaluate
        """
        raise NotImplementedError()
    
    @classmethod
    # @abstractmethod
    def plot_training(self, rewards, mean_window,method,dirname) -> None:
        
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(10,4.5), dpi= 120, facecolor='w', edgecolor='k')
        ax1.plot(range(0,len(rewards)),rewards)
        ax2.plot(range(self.mean_window-1,len(rewards)), np.convolve(rewards, np.ones(mean_window)/mean_window, mode='valid'),c='m')
        ax1.set_title("Rewards vs Episode",fontweight='bold',fontsize = 11)
        ax2.set_title("{} Avg Rewards vs Episode".format(mean_window),fontweight='bold',fontsize = 11)
        fig.suptitle('Training Performance\n\n',fontweight='bold',fontsize = 14)
        ax1.set_xlabel('Episode',fontweight='bold',fontsize = 8)
        ax1.set_ylabel('Episode Rewards',fontweight='bold',fontsize = 8)
        ax2.set_xlabel('Last Episode',fontweight='bold',fontsize = 8)
        ax2.set_ylabel('Mean 100 Rewards',fontweight='bold',fontsize = 8)
        ax1.grid()
        ax2.grid()
        fig.savefig(dirname + 'Plots/' + method + '_Training_' + time.strftime("%Y%m%d-%H%M%S") + '.png')
        plt.pause(0.001)

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

    def length(self):
        return len(self.is_done)