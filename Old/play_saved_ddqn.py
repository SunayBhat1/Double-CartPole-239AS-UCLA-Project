from carts_poles import CartsPolesEnv
from DDQN_pytorch import QNetwork
import torch
import numpy as np
import matplotlib.pyplot as plt

model=torch.load('DQN_max.pt')

env = CartsPolesEnv()
layer_one=model['fc_1.weight'].shape
layer_two=model['fc_2.weight'].shape
layer_three=model['fc_3.weight'].shape
Qmodel=QNetwork(layer_three[0],layer_one[1],layer_one[0])
Qmodel.load_state_dict(model)

history = list()
state = env.reset()
done = False
total_reward = 0
prev = 0
x = range(200)
x2 = []
y = []
for i in x:
    total_reward = 0
    stateVal = ((i-100)/8*np.pi/100)
    x2.append(stateVal)
    print(stateVal)
    #print(stateVal)
    state = env.reset(stateVal)
    done = False
    while not done:
        state=torch.Tensor(state)
        with torch.no_grad():
                values = Qmodel(state)
        action = np.argmax(values.cpu().numpy())
        state, reward, done, info= env.step(action)
        # history.append((s,a,r))
        #env.render()
        total_reward += reward
        if(info['time']-prev>1):
            #print(info['time'])
            prev = info['time']
        if info['time']>200:
            done=True
    y.append(info['time'])
#env.close()
plt.plot(x2,y)
plt.show()