import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque
import time
import random
import cv2

SNAKE_MAX_LENGTH = 30


def collision_with_apple(apple_position, score):
    score += 1
    return apple_position, score

def collision(snake_position):
    snake_head = snake_position[0]
    if snake_head[0]>=500 or snake_head[0]<0 or snake_head[1]>=500 or snake_head[1]<0 :
        return 1
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0    


class SnakeEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(4)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-500, high=500,
                                            shape=(5 + SNAKE_MAX_LENGTH,), dtype=np.float32)

    def step(self, action):
        self.prev_moves.append(action)
        self.reward = 0
        cv2.imshow('a', self.img)
        cv2.waitKey(1)
        self.img = np.zeros((500,500,3),dtype='uint8')
        # Display Apple
        cv2.rectangle(self.img,(self.apple_position[0],self.apple_position[1]),(self.apple_position[0]+10,self.apple_position[1]+10),(0,0,255),3)
        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(self.img,(position[0],position[1]),(position[0]+10,position[1]+10),(0,255,0),3)
        
        # Takes step after fixed time
        t_end = time.time() + 0.01
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(125)
            else:
                continue
                
        # Change the head position based on the action
        if action == 0:
            self.snake_head[0] += 10
        elif action == 1:
            self.snake_head[0] -= 10
        elif action == 2:
            self.snake_head[1] += 10
        elif action == 3:
            self.snake_head[1] -= 10

        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
            self.score += 1
            self.snake_position.insert(0,list(self.snake_head))
        else:
            self.snake_position.insert(0,list(self.snake_head))
            self.snake_position.pop()
        
        # On collision kill the snake and print the score
        if collision(self.snake_position) == 1:
            font = cv2.FONT_HERSHEY_SIMPLEX
            self.img = np.zeros((500,500,3),dtype='uint8')
            cv2.putText(self.img,'Your Score is {}'.format(self.score),(140,250), font, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow('a',self.img)
            self.done = True

        
        # Update the Observation Space
        # head_x, head_y, apple_dist_x, apple_dist_y, snake_length, prev_moves 
        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        apple_dist_x = head_x - self.apple_position[0]
        apple_dist_y = head_y - self.apple_position[1]

        snake_length = len(self.snake_position)

        self.observation = [head_x, head_y, apple_dist_x, apple_dist_y, snake_length] + list(self.prev_moves)
        self.observation = np.array(self.observation, dtype=np.float32)

        terminated = self.done
        truncated = self.done
        info = {}

        # Compute Reward
        prev_apple_dist_x = self.prev_head[0] - self.apple_position[0]
        prev_apple_dist_y = self.prev_head[1] - self.apple_position[1]
        if self.done: self.reward -= 10
        else:
            if self.score > self.prev_score:
                self.reward += 10
            self.reward += -0.01 * (abs(apple_dist_x) - abs(prev_apple_dist_x)) + -0.01 * (abs(apple_dist_y) - abs(prev_apple_dist_y))
            self.prev_score = self.score

            
        return self.observation, self.reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.done = False
        self.img = np.zeros((500,500,3),dtype='uint8')

        # Initial Snake and Apple position
        self.snake_position = [[250,250],[240,250],[230,250]]
        self.apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
        self.reward = 0
        self.score = 0
        self.prev_score = 0 
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [250,250]
        self.prev_head = [250,250]

        # Observation Space
        # head_x, head_y, apple_dist_x, apple_dist_y, snake_length, prev_moves 
        head_x = self.snake_head[0]
        head_y = self.snake_head[1]

        apple_dist_x = head_x - self.apple_position[0]
        apple_dist_y = head_y - self.apple_position[1]

        snake_length = len(self.snake_position)

        self.prev_moves = deque(maxlen=SNAKE_MAX_LENGTH)

        for _ in range(SNAKE_MAX_LENGTH):
            self.prev_moves.append(-1)

        self.observation = [head_x, head_y, apple_dist_x, apple_dist_y, snake_length] + list(self.prev_moves)
        self.observation = np.array(self.observation, dtype=np.float32)

        self.info = {}

        return self.observation, self.info

    #def render(self):
    #    ...

    #def close(self):
    #    ...