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
    def load(self, dirname: str, file_ext: str) -> None:
        """ Load in the network for this class
        """
        raise NotImplementedError()
     
    @abstractmethod
    def evaluate(self, dirname: str, plot: bool) -> float:
        """ 
        Return the performance of the agent over all the angles in the range
        """
        raise NotImplementedError()

    @abstractmethod
    def run_training(self, dirname: str, print_log: int) -> None:
        """ Main loop, returns trained agent
            save agents that pass evaluate
        """
        raise NotImplementedError()

    @abstractmethod
    def render_run(self,iters: int) -> None:
        """ 
        Run a single episode with render (random angle)
        """
        raise NotImplementedError()
    

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