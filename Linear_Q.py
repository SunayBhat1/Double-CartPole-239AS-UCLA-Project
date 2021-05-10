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
NUM_EPISODES = 10000
MAX_T = 1000
GAMMA = 0.7
ALPHA = 0.1
E_GREEDY = 0.1

# Function approx to compute (makes weights 4 sets of states, and turns all off and on based on which action)
def Q_value(state,action,linear_weights):

    x = np.zeros(20)
    x[action*10:action*10+10] = state

    return linear_weights @ x

def get_features(state,action):

    x = np.zeros(20)
    x[action*10:action*10+10] = state

    return x

# Get action e-greedy
def get_action(linear_weights, state):
    p_epsilon = np.random.uniform(0,1)
    if p_epsilon < E_GREEDY:
        return np.argmax(np.random.uniform(0,1,(1,2)))

    q_s = np.zeros(2)

    for i in range(0,2):
        q_s[i] = Q_value(state,i,linear_weights)

    return np.argmax(q_s)


# Plotting Stuff
ep_length = np.zeros(NUM_EPISODES)
td_error = np.zeros(NUM_EPISODES)


# Lets get some linear weights
linear_weights = np.random.rand(1,20)

# init env
env = CartsPolesEnv()

done = False

for episode in range(NUM_EPISODES):

    env.reset()
    done = False

    state,_,_ = env.step(env.dt,1)

    action = get_action(linear_weights, state)

    error_episode = 0

    # Generate an episode
    for t in range(MAX_T):

        action_prime = get_action(linear_weights,state)

        state_prime, reward, done = env.step(env.dt,action_prime)

        # Linear SARSA update (Section 10.1, psuedocode) 
        td_update = (reward + GAMMA * Q_value(state_prime,action_prime,linear_weights)-Q_value(state,action,linear_weights)) * get_features(state,action)      
        linear_weights = linear_weights + ALPHA*td_update

        error_episode += np.sum(td_update)

        state  = state_prime
        action = action_prime

        if done or t == MAX_T - 1:
            ep_length[episode] = t
            td_error[episode] = error_episode
            break

    if episode % 1000 == 0:
        print("Episode %d completed, avge steps now is  %d steps" % (episode, ep_length.mean()))


fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12,4), dpi= 100, facecolor='w', edgecolor='k')
ax1.plot(range(0,NUM_EPISODES),ep_length)
ax2.plot(range(0,NUM_EPISODES),td_error,c='g')
ax1.title.set_text("Episode Length vs Episode")
ax2.title.set_text("TD Error Convergence")
fig.suptitle('SARSA TD For Q-Value Control')
# ax1.set_xscale('log')
# ax2.set_xscale('log')
ax1.grid()
ax2.grid()
plt.show()

print(linear_weights)