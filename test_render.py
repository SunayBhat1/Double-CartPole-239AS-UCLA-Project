import gym
from carts_poles import CartsPolesEnv
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time
 


# init env
env = CartsPolesEnv()

state_prime, reward, done, _= env.step(1)
env.render()

# Wait for 5 seconds
time.sleep(5)