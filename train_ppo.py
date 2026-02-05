from SnakeEnv import SnakeEnv
from stable_baselines3 import PPO

env = SnakeEnv()
env.reset()
model = PPO("MlpPolicy", env, verbose = 1)

model.learn(total_timesteps = int(2e5), progress_bar = True)

model.save("SnakeModel")