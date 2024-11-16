import numpy as np
import random
from collections import defaultdict

# class Q_Learning(object):

#     def __init__(self,car_pos, car_angle, car_speed, collision):
#         self.car_angle = car_angle
#         self.car_pos = car_pos
#         self.car_speed = car_speed
#         self.collision = collision

#     def get_state(self):
#         pos_x, pos_y = self.car_pos
#         angle = int(self.car_angle) % 360
#         speed = int(self.car_speed)
#         return (int(pos_x // 40), int(pos_y // 40), angle, speed)

#     # Define rewards including distance-based reward
#     def compute_reward(self):
#         if self.collision:
#             return -10  # Penalize for collision
        
#         # Distance reward: encourage staying in the middle of the track
#         # distance_reward = calculate_distance_reward()
#         return self.car_speed  # Base reward + distance-based reward

#     # Q-learning Action Selection (ε-greedy)
#     def choose_action(self, state):
#         # ------------------------------------------------------------ config -------------------------------------------------------
#         epsilon = 0.1  # Exploration factor

#         if random.uniform(0, 1) < epsilon:
#             return random.randint(0, 3)  # Explore: random action
#         return np.argmax(Q[state])  # Exploit: best known action

#     # Update Q-table
#     def update_q_table(state, action, reward, next_state):
#         #   ------------------------------------------------------------ config -------------------------------------------------------
#         Q = defaultdict(lambda: np.zeros(4))  # Q-table with 4 actions: 0=accelerate, 1=brake, 2=turn left, 3=turn right
#         alpha = 0.1  # Learning rate
#         gamma = 0.9  # Discount factor

#         best_next_action = np.argmax(Q[next_state])
#         td_target = reward + gamma * Q[next_state][best_next_action]
#         td_error = td_target - Q[state][action]
#         Q[state][action] += alpha * td_error
