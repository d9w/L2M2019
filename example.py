from osim.env import L2M2019Env
import numpy as np

env = L2M2019Env(visualize=False)
np.random.seed(0) # deterministic vector field
observation = env.reset()
for i in range(200):
    observation, reward, done, info = env.step(env.action_space.sample())
    print(i, reward, done, info)
