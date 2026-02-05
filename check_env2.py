from SnakeEnv import SnakeEnv
import cv2

env = SnakeEnv()
ep = 50

for e in range(ep):
    done = False
    obs = env.reset()
    while not done:
        random_action = 0
        print("action: ", random_action)
        obs, reward, done, _, _ = env.step(random_action)
        print("reward: ", reward)
