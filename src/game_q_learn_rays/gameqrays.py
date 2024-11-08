import pygame
import numpy as np
import random
from collections import defaultdict
import math
import pickle
import os

# Initialize Pygame
pygame.init()

# Screen Settings
SCREEN_WIDTH, SCREEN_HEIGHT = 1528, 798
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("2D AI Racing Game")
clock = pygame.time.Clock()

# Define Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

track_image = pygame.image.load("track2.png").convert()

# Car Settings
car_img = pygame.Surface((20, 10), pygame.SRCALPHA)
pygame.draw.polygon(car_img, RED, [(0, 0), (20, 5), (0, 10)])  # Triangle shape

# Car properties
start_pos = [200, 100]  # Start position within the track
car_pos = start_pos[:]
car_angle = 0
car_speed = 2
MAX_SPEED = 5
ACCELERATION = 0.2
TURN_ANGLE = 5

# ____________________________________________________________ Q_Learning ___________________________________________________________________________________________

# Q-learning Parameters
Q = defaultdict(lambda: np.zeros(4))  # Q-table with 4 actions: 0=accelerate, 1=brake, 2=turn left, 3=turn right
epsilon = 0.1  # Exploration factor
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor


# Define discrete states for Q-learning
def get_state():
    pos_x, pos_y = car_pos
    angle = int(car_angle) % 360
    speed = int(car_speed)
    rays = get_ray_distances(car_pos, car_angle, num_rays=8, ray_length=300)
    return tuple(rays) + (speed, ACCELERATION)
    # return tuple(rays) + (int(pos_x // 40), int(pos_y // 40), angle, speed)


# Define rewards including distance-based reward
def compute_reward(i):
    if check_collision():
        return -100  # Penalize for collision

    # Distance reward: encourage staying in the middle of the track
    # distance_reward = calculate_distance_reward()
    return (car_speed/MAX_SPEED) * i  # Base reward + distance-based reward

# Q-learning Action Selection (ε-greedy)
def choose_action(state):
    state = tuple(state)
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)  # Explore: random action
    return np.argmax(Q[state])  # Exploit: best known action

# Update Q-table
def update_q_table(state, action, reward, next_state):
    state = tuple(state)
    next_state = tuple(state)
    best_next_action = np.argmax(Q[next_state])
    td_target = reward + gamma * Q[next_state][best_next_action]
    td_error = td_target - Q[state][action]
    Q[state][action] += alpha * td_error

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Save Q-table to a file
def save_q_table(filename="q_table.pkl"):
    try:
        with open(filename, "wb") as file:
            pickle.dump(dict(Q), file)
        print("Data saved successfully.")
    except Exception as e:
        print("Error saving data:", e)


# Load Q-table from a file
def load_q_table(filename="q_table.pkl"):
    print("File size:", os.path.getsize(filename), "bytes")
    with open(filename, "rb") as file:
        q_table = pickle.load(file)
    q_table = defaultdict(lambda: np.zeros(4), q_table)
    # print(q_table)          # Display the entire Q table
    # print(q_table.keys())   # Display all keys (states)
    # print(q_table[0])
    return q_table

# ________________________________________________________________________________________________________________________________________________________________________________

# Move car with acceleration and rotation
def move_car(action):
    global car_angle, car_speed
    
    # Actions: 0=accelerate, 1=brake, 2=turn left, 3=turn right
    if action == 0:  # Accelerate
        car_speed = min(car_speed + ACCELERATION, MAX_SPEED)
    elif action == 1:  # Brake
        car_speed = max(car_speed - ACCELERATION, -MAX_SPEED) #actual brake
        # car_speed = min(car_speed + ACCELERATION, MAX_SPEED) #only speed
    elif action == 2:  # Turn left
        car_angle += TURN_ANGLE
    elif action == 3:  # Turn right
        car_angle -= TURN_ANGLE

    # Update car position based on current speed and angle
    rad = np.deg2rad(car_angle)
    car_pos[0] += car_speed * np.cos(rad)
    car_pos[1] += car_speed * np.sin(rad)

# Check for collisions with boundary
def check_collision():
    car_rect = car_img.get_rect(center=(car_pos[0], car_pos[1]))
    track_color_at_car = screen.get_at((int(car_rect.centerx), int(car_rect.centery)))

    #Black is the track color
    if track_color_at_car != BLACK:
        return True
    return False

# ------------------------------------ Rays -------------------------------------

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
        ray_distances.append(distance / ray_length)  # Normalize distance to [0, 1]

        # draw_ray(car_pos, distance, angle)
    return ray_distances


# Q = load_q_table()

# Main Game Loop for Q-learning Training
for episode in range(100):  # Train for a number of episodes
    car_pos = start_pos[:]
    car_angle = 0
    car_speed = 2
    total_reward = 0
    i = 0

    while True:
        i += 0.1
        # Handle events to keep Pygame responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        #draw screen
        screen.blit(track_image, (0, 0))

        # Check if episode is over (collision or timeout)
        if check_collision() or total_reward > 1000:
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")
            break

        # Get current state
        state = get_state()
        
        # Choose an action based on current state
        action = choose_action(state)

        # Move car based on action
        move_car(action)
        
        # Compute reward and check for collisions
        reward = compute_reward(i)

        total_reward += reward

        # Get next state
        next_state = get_state()

        # Update Q-table
        update_q_table(state, action, reward, next_state)

        #draw car
        rotated_car = pygame.transform.rotate(car_img, -car_angle)
        car_rect = rotated_car.get_rect(center=(car_pos[0], car_pos[1]))

        screen.blit(rotated_car, car_rect.topleft)
        
        pygame.display.flip()
        clock.tick(300)

save_q_table()