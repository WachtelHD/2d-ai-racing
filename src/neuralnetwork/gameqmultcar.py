import pygame
import numpy as np
import random
from collections import defaultdict
import math
import pickle
import os
import cv2
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras import layers, models

# Initialize Pygame
pygame.init()
pygame.display.set_caption("2D AI Racing Game")
clock = pygame.time.Clock()

# Screen Settings
SCREEN_WIDTH, SCREEN_HEIGHT = 1528, 798
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Define Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Track
track_image_path = "track2.png"
track_image = pygame.image.load(track_image_path).convert()

# Car Settings
car_img = pygame.Surface((20, 10), pygame.SRCALPHA)
pygame.draw.polygon(car_img, RED, [(0, 0), (20, 5), (0, 10)])  # Triangle shape

# Car properties
START_POS = [200, 100]  # Start position within the track
START_SPEED = 2
START_ANGLE = 15

# Number of cars
NUM_CARS = 1
car_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(NUM_CARS)]

# Initialize cars with separate positions, speeds, and angles
car_positions = [START_POS[:] for _ in range(NUM_CARS)]
car_angles = [START_ANGLE for _ in range(NUM_CARS)]
car_speeds = [START_SPEED for _ in range(NUM_CARS)]

MAX_SPEED = 5
ACCELERATION = 0.2
TURN_ANGLE = 5

replay_buffers = [deque(maxlen=10000) for _ in range(NUM_CARS)] # Stores up to 10,000 experiences
BATCH_SIZE = 50

# -------------------------- Casting Rays -----------------------------------

def cast_ray_skeleton(car_pos, angle, max_distance=900):
    pos_x, pos_y = car_pos
    distance = 0
    step_size = 2
    while distance < max_distance:
        x = int(pos_x + distance * math.cos(math.radians(angle)))
        y = int(pos_y + distance * math.sin(math.radians(angle)))
        
        # Ensure we stay within screen bounds
        if x < 0 or x >= SCREEN_WIDTH or y < 0 or y >= SCREEN_HEIGHT:
            return max_distance  # Return max distance if out of bounds
        
        # Check the color at this position
        color = skeleton.get_at((x, y))
        if color != BLACK:
            return distance  # Collision detected at this distance
        
        distance += step_size
    
    return max_distance  # No collision within max distance

def draw_ray_skeleton(car_pos, distance, angle):
    pos_x, pos_y = car_pos

    # Calculate the end point of the ray based on the detected distance
    ray_end_x = pos_x + distance * math.cos(math.radians(angle))
    ray_end_y = pos_y + distance * math.sin(math.radians(angle))
    
    # Draw the ray
    pygame.draw.line(screen, (255, 255, 0), car_pos, (ray_end_x, ray_end_y), 1)
    
    # Optionally draw a small circle at the end of the ray
    pygame.draw.circle(screen, (0, 255, 0), (int(ray_end_x), int(ray_end_y)), 3)

def get_ray_distances_skeleton(car_pos, car_angle, num_rays=14, ray_length=100):
    ray_distances = []
    ray_angles = [car_angle + i * (360 / num_rays) for i in range(num_rays)]
    for angle in ray_angles:
        distance = cast_ray_skeleton(car_pos, angle, ray_length)
        ray_distances.append(distance / ray_length)  # Normalize distance to [0, 1]

        # draw_ray_skeleton(car_pos, distance, angle)
    return ray_distances

def cast_ray(car_pos, angle, max_distance=900):
    pos_x, pos_y = car_pos
    distance = 0
    step_size = 5
    while distance < max_distance:
        x = int(pos_x + distance * math.cos(math.radians(angle)))
        y = int(pos_y + distance * math.sin(math.radians(angle)))
        
        # Ensure we stay within screen bounds
        if x < 0 or x >= SCREEN_WIDTH or y < 0 or y >= SCREEN_HEIGHT:
            return max_distance  # Return max distance if out of bounds
        
        # Check the color at this position
        color = track_image.get_at((x, y))
        if color != BLACK:
            return distance  # Collision detected at this distance
        
        distance += step_size
    
    return max_distance  # No collision within max distance

def draw_ray(car_pos, distance, angle):
    pos_x, pos_y = car_pos

    # Calculate the end point of the ray based on the detected distance
    ray_end_x = pos_x + distance * math.cos(math.radians(angle))
    ray_end_y = pos_y + distance * math.sin(math.radians(angle))
    
    # Draw the ray
    pygame.draw.line(screen, (255, 255, 0), car_pos, (ray_end_x, ray_end_y), 1)
    
    # Optionally draw a small circle at the end of the ray
    pygame.draw.circle(screen, (0, 255, 0), (int(ray_end_x), int(ray_end_y)), 3)

def get_ray_distances(car_pos, car_angle, num_rays=20, ray_length=900):
    ray_distances = []
    ray_angles = [car_angle + i * (360 / num_rays) for i in range(num_rays)]
    for angle in ray_angles:
        distance = cast_ray(car_pos, angle, ray_length)
        ray_distances.append(distance)  # Normalize distance to [0, 1]

        draw_ray(car_pos, distance, angle)
    return ray_distances


# --------------------------- Buffer ----------------------------------------

def store_experience(state, action, reward, next_state, buffer_num):
    experience = (state, action, reward, next_state)
    replay_buffers[buffer_num].append(experience)

def train_from_replay(replay_buffer):
    batch = random.sample(replay_buffer, min(len(replay_buffer), BATCH_SIZE))


    for state, action, reward, next_state in batch:
        # Update Q-table based on the sampled experience
        best_next_action = np.argmax(Q[next_state])  # Find the best action in the next state
        td_target = reward + gamma * Q[next_state][best_next_action]
        td_error = td_target - Q[state][action]
        Q[state][action] += alpha * td_error


# ---------------------------- Drawing ---------------------------------------

def draw_cars():
    for i in range(NUM_CARS):
        rotated_car = pygame.transform.rotate(car_img, -car_angles[i])
        car_rect = rotated_car.get_rect(center=(car_positions[i][0], car_positions[i][1]))
        screen.blit(rotated_car, car_rect.topleft)

        # Draw car using the car's unique color
        car_surface = pygame.Surface((20, 10), pygame.SRCALPHA)
        pygame.draw.polygon(car_surface, car_colors[i], [(0, 0), (20, 5), (0, 10)])  # Triangle shape
        rotated_surface = pygame.transform.rotate(car_surface, -car_angles[i])
        screen.blit(rotated_surface, car_rect.topleft)

def generate_skeleton():
    # Load the track image and preprocess it
    track_image_gray = cv2.imread(track_image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to binary
    _, binary_track = cv2.threshold(track_image_gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Skeletonize the binary track image to get the centerline
    skeleton = cv2.ximgproc.thinning(binary_track)

    cv2.imwrite("skeleton_with_checkpoints.png", skeleton)

# generate_skeleton()

skeleton = pygame.image.load("skeleton_with_checkpoints.png").convert()

# ---------------------------- Car Information -------------------------------

# Define discrete states for Q-learning
def get_state(car_pos, car_angle, car_speed):
    # pos_x, pos_y = car_pos
    # angle = int(car_angle) % 360
    # speed = int(car_speed)
    rays = get_ray_distances(car_pos, car_angle, num_rays=8, ray_length=100)
    return np.array(rays)
    # return tuple(rays) + (int(pos_x // 40), int(pos_y // 40), angle, speed)

# Move car with acceleration and rotation
def move_car(car_num, action):    
    # Actions: 0=accelerate, 1=brake, 2=turn left, 3=turn right
    if action == 0:  # Accelerate
        car_speeds[car_num] = min(car_speeds[car_num] + ACCELERATION, MAX_SPEED)
    elif action == 1:  # Brake
        car_speeds[car_num] = max(car_speeds[car_num] - ACCELERATION, -MAX_SPEED) #actual brake
        # car_speed = min(car_speed + ACCELERATION, MAX_SPEED) #only speed
    elif action == 2:  # Turn left
        car_angles[car_num] += TURN_ANGLE
    elif action == 3:  # Turn right
        car_angles[car_num] -= TURN_ANGLE

    # Update car position based on current speed and angle
    rad = np.deg2rad(car_angles[car_num])
    car_positions[car_num][0] += car_speeds[car_num] * np.cos(rad)
    car_positions[car_num][1] += car_speeds[car_num] * np.sin(rad)

# Check for collisions with boundary
def check_collision(car_position):
    x, y = int(car_position[0]), int(car_position[1])
    if x < 0 or x >= SCREEN_WIDTH or y < 0 or y >= SCREEN_HEIGHT:
        return True  # Out of bounds
    color = track_image.get_at((x, y))
    return color != BLACK  # Collision if not on track


# ------------------------------------ Neural Network --------------------------------

EPISODES = 1000
MAX_STEPS = 200
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1


# !!!!!!!!!!!!!!! Use conda !!!!!!!!!!!!!!

# Neural network model
def create_model(input_size, output_size):
    model = models.Sequential([
        layers.Dense(8, activation='relu', input_shape=(input_size,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(output_size, activation='softmax')  # Softmax for classification
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Instantiate model
INPUT_SIZE = 8  # 8 rays 
                # + 2 velocity components
OUTPUT_SIZE = 4  # 4 discrete actions
model = create_model(INPUT_SIZE, OUTPUT_SIZE)
model.summary()

def compute_reward(car_pos, car_angle, car_speed, steps_alive):
    if check_collision(car_pos):
        return -1000  # Penalize for collision

    reward = 0

    min_skeleton_rays = min(get_ray_distances_skeleton(car_pos, car_angle))

    if min_skeleton_rays < 0.15:
        reward += 1

    if min_skeleton_rays < 0.1:
        reward += 2

    if min_skeleton_rays < 0.05:
        reward += 4

    # Small reward based on the car's speed to encourage faster movement
    if (car_speed / MAX_SPEED) > 0.7:
        reward += (car_speed / MAX_SPEED) * 2
    if (car_speed / MAX_SPEED) < 0.1:
        reward -= 20

    # reward += (steps_alive/10)

    return reward/2

# ------------------------------------ Game ------------------------------------------

# Q = load_q_table()

total_rewards = [0] * NUM_CARS  # Track rewards for each car
steps_alive = [0] * NUM_CARS  # Track steps alive for each car

# Main Game Loop for Q-learning Training
for episode in range(50):  # Train for a number of episodes2
    cars_run = 0

    while True:

        # Handle events to keep Pygame responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        #draw screen
        screen.blit(track_image, (0, 0))

        for i in range(NUM_CARS):
            steps_alive[i] += 1
            
            # Get current state for each car
            state = get_state(car_positions[i], car_angles[i], car_speeds[i])

            # if state.ndim == 1:  # If state is 1D, add batch dimension
            #     state = state[np.newaxis, :]

            # # Choose an action based on the current state
            # action = choose_action(state)

            # Choose action (epsilon-greedy)
            if random.random() < EPSILON:
                action = random.randint(0, OUTPUT_SIZE - 1)  # Random action
            else:
                action = np.argmax(model.predict(state[np.newaxis])[0])  # Predict best action

            # Move each car independently
            move_car(i, action)

            # Compute reward and check for collisions
            reward = compute_reward(car_positions[i], car_angles[i], car_speeds[i], steps_alive[i])

            total_rewards[i] += reward
            # Get next state after the move
            next_state = get_state(car_positions[i], car_angles[i], car_speeds[i])
            # if next_state.ndim == 1:  # If state is 1D, add batch dimension
            #     next_state = next_state[np.newaxis, :]

            target = reward

            if check_collision(car_positions[i]):
                target += GAMMA * np.max(model.predict(next_state)[0])
                
            target_f = model.predict(state[np.newaxis, :])
            target_f[0][action] = reward + GAMMA * np.max(model.predict(next_state[np.newaxis, :])[0])
            model.fit(state[np.newaxis, :], np.array([action]), verbose=0)

            
            # # Store experience in replay buffer
            # store_experience(state, action, reward, next_state, i)
            
            # # Update Q-table using replay if buffer has enough experiences
            # if len(replay_buffers[i]) > BATCH_SIZE:
            #     train_from_replay(replay_buffers[i])

            # Check if the car collided or reached the max reward
            # if check_collision(car_positions[i]) or total_rewards[i] > 50000:
            
            if check_collision(car_positions[i]):
                    print(f"Car {i} - Episode {episode + 1}: Total Reward: {total_rewards[i]}")
                    # Reset the car's position, angle, and speed for the next episode
                    car_positions[i] = START_POS[:]
                    car_angles[i] = START_ANGLE
                    car_speeds[i] = START_SPEED
                    total_rewards[i] = 0   # Track rewards for each car
                    steps_alive[i] = 0  # Track steps alive for each car
                    total_rewards[i] = 0
                    steps_alive[i] = 0
                    cars_run += 1
            
            # Draw all cars on the track
        draw_cars()

        pygame.display.flip()
        clock.tick(300)

        if cars_run > 50:
            break

# save_q_table()