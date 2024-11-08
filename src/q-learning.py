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

# def calculate_distance_reward():
#     # Calculate the distance from the car to both inner and outer boundaries
#     car_rect = pygame.Rect(car_pos[0], car_pos[1], 1, 1)
    
#     # Distance to outer boundary (calculate minimum distance to edges)
#     outer_left = abs(car_pos[0] - OUTER_BOUNDARY_RECT.left)
#     outer_right = abs(OUTER_BOUNDARY_RECT.right - car_pos[0])
#     outer_top = abs(car_pos[1] - OUTER_BOUNDARY_RECT.top)
#     outer_bottom = abs(OUTER_BOUNDARY_RECT.bottom - car_pos[1])
#     distance_to_outer = min(outer_left, outer_right, outer_top, outer_bottom)
    
#     # Distance to inner boundary (calculate minimum distance to edges)
#     inner_left = abs(car_pos[0] - INNER_BOUNDARY_RECT.left)
#     inner_right = abs(INNER_BOUNDARY_RECT.right - car_pos[0])
#     inner_top = abs(car_pos[1] - INNER_BOUNDARY_RECT.top)
#     inner_bottom = abs(INNER_BOUNDARY_RECT.bottom - car_pos[1])
#     distance_to_inner = min(inner_left, inner_right, inner_top, inner_bottom)
    
#     # Calculate reward based on the minimum distance to the boundaries
#     max_distance = min(OUTER_BOUNDARY_RECT.width // 2, OUTER_BOUNDARY_RECT.height // 2)
#     distance_from_boundaries = min(distance_to_outer, distance_to_inner)
#     distance_reward = (distance_from_boundaries / max_distance)  # Normalize by max distance
    
#     return distance_reward  # This reward is between 0 and 1