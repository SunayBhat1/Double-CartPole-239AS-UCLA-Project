import gym
from carts_poles import CartsPolesEnv
import numpy as np
import random
import math
import matplotlib.pyplot as plt


# init env
env = CartsPolesEnv()

state_prime, reward, done = env.step(1)
env.render()