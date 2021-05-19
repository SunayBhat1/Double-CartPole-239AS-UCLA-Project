from carts_poles import CartsPolesEnv
from DQN_pytorch import Agent
from DQN_pytorch import DQN
import torch
import numpy as np


env = CartsPolesEnv()
model=torch.load('DQN_19_05_2021_12_17_33.pt')
layer_one=model['layer1.0.weight'].shape
layer_two=model['layer2.0.weight'].shape
final_layer=model['final.weight'].shape
print(layer_two)
print(layer_one)
print(final_layer)
agent = Agent(layer_one[1], final_layer[0], layer_two[0])
agent.dqn.load_state_dict(model)

history = list()
s = env.reset()
done = False
total_reward = 0

while not done:
    a = agent.get_action(s, 0)
    s2, r, done, info = env.step(a)
    # history.append((s,a,r))
    env.render()
    total_reward += r
    if done:
        r = -1
    s = s2
env.close()