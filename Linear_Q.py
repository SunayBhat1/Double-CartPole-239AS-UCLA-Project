import gym
from carts_poles import CartsPolesEnv
import numpy as np
import random
import math
import matplotlib.pyplot as plt

"""
Resources:
https://www.andrew.cmu.edu/course/10-403/slides/S19_lecture8_FApredictioncontrol.pdf
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
https://github.com/ankitdhall/CartPole/blob/master/Qlearning-linear.py
"""

#Hyperparms
NUM_EPISODES = 10
MAX_T = 1000
GAMMA = 0.9
ALPHA = 0.1
E_GREEDY = 0.1

# Function approx to compute (makes weights 4 sets of states, and turns all off and on based on which action)
def Q_value(state,action,linear_weights):

    x = np.zeros(40)
    x[action*10:action*10+10] = state

    return linear_weights @ x

def get_features(state,action):

    x = np.zeros(40)
    x[action*10:action*10+10] = state

    return x

# Get action e-greedy
def get_action(linear_weights, state):
    p_epsilon = np.random.uniform(0,1)
    if p_epsilon < E_GREEDY:
        return np.argmax(np.random.uniform(0,1,(1,4)))

    q_s = np.zeros(4)

    for i in range(0,4):
        q_s[i] = Q_value(state,i,linear_weights)

    return np.argmax(q_s)

    


# Lets get some linear weights
linear_weights = np.random.rand(1,40)

# init env
env = CartsPolesEnv()

done = False

sample_count = 0

for episode in range(NUM_EPISODES):

    env.reset()
    done = False

    sample_count += 1

    state,_,_ = env.step(env.dt,2)

    action = get_action(linear_weights, state)

    # Generate an episode
    for t in range(MAX_T):

        action_prime = get_action(linear_weights,state)

        state_prime, reward, done = env.step(env.dt,action_prime)

        # Linear SARSA update (Section 10.1, psuedocode)
        print(reward + GAMMA * Q_value(state_prime,action_prime,linear_weights)-Q_value(state,action,linear_weights))
        linear_weights = linear_weights + ALPHA*(reward + GAMMA * Q_value(state_prime,action_prime,linear_weights)-Q_value(state,action,linear_weights)) * get_features(state,action)

        # linear_weights = linear_weights-np.min(linear_weights)
        # linear_weights = linear_weights/np.max(linear_weights)

        state  = state_prime
        action = action_prime

        if done or t == MAX_T - 1:
            print("Episode %d completed in %d steps" % (sample_count, t))
            plt.scatter(sample_count,t)
            break

    # plt.plot(linear_weights)

    # print(np.min(linear_weights),np.max(linear_weights))
plt.title("Episode Lengths vs Iteration")
plt.show()

print(linear_weights)