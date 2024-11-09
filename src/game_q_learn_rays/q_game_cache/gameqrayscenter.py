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


# Define the replay buffer with a maximum size
replay_buffer = deque(maxlen=10000)  # Stores up to 10,000 experiences
batch_size = 50

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
start_pos = [200, 100]  # Start position within the track
car_pos = start_pos[:]
car_angle = 20
car_speed = 2

# Number of cars
NUM_CARS = 5

# Initialize cars with separate positions, speeds, and angles
car_positions = [start_pos[:] for _ in range(NUM_CARS)]
car_angles = [20 for _ in range(NUM_CARS)]
car_speeds = [2 for _ in range(NUM_CARS)]

MAX_SPEED = 5
ACCELERATION = 0.2
TURN_ANGLE = 5


def store_experience(state, action, reward, next_state):
    experience = (state, action, reward, next_state)
    replay_buffer.append(experience)

def sample_experiences(batch_size=32):
    # Sample a random mini-batch from the replay buffer
    batch = random.sample(replay_buffer, min(len(replay_buffer), batch_size))
    return batch

def train_from_replay():
    batch = sample_experiences()

    for state, action, reward, next_state in batch:
        # Update Q-table based on the sampled experience
        best_next_action = np.argmax(Q[next_state])  # Find the best action in the next state
        td_target = reward + gamma * Q[next_state][best_next_action]
        td_error = td_target - Q[state][action]
        Q[state][action] += alpha * td_error

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def generate_skeleton():
    # Load the track image and preprocess it
    track_image_gray = cv2.imread(track_image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to binary
    _, binary_track = cv2.threshold(track_image_gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Skeletonize the binary track image to get the centerline
    skeleton = cv2.ximgproc.thinning(binary_track)

    cv2.imwrite("skeleton_with_checkpoints.png", skeleton)

skeleton = pygame.image.load("skeleton_with_checkpoints.png").convert()

def cast_ray_skeleton(car_pos, angle, max_distance=900):
    pos_x, pos_y = car_pos
    distance = 0
    step_size = 1
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

def get_ray_distances_skeleton(car_pos, car_angle, num_rays=20, ray_length=300):
    ray_distances = []
    ray_angles = [car_angle + i * (360 / num_rays) for i in range(num_rays)]
    for angle in ray_angles:
        distance = cast_ray_skeleton(car_pos, angle, ray_length)
        ray_distances.append(distance / ray_length)  # Normalize distance to [0, 1]

        # draw_ray_skeleton(car_pos, distance, angle)
    return ray_distances

# ____________________________________________________________ Q_Learning ___________________________________________________________________________________________

# Q-learning Parameters
Q = defaultdict(lambda: np.zeros(4))  # Q-table with 4 actions: 0=accelerate, 1=brake, 2=turn left, 3=turn right
epsilon = 0  # Exploration factor
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor

# Define discrete states for Q-learning
def get_state():
    pos_x, pos_y = car_pos
    angle = int(car_angle) % 360
    speed = int(car_speed)
    rays = get_ray_distances(car_pos, car_angle, num_rays=8, ray_length=150)
    return tuple(rays) + (speed, ACCELERATION)
    # return tuple(rays) + (int(pos_x // 40), int(pos_y // 40), angle, speed)

# Define rewards including distance-based reward
def compute_reward(steps_alive):
    if check_collision():
        return -1000  # Penalize for collision

    reward = 0

    if min(get_ray_distances_skeleton(car_pos, car_angle)) < 0.15:
        reward += 0.5

    if min(get_ray_distances_skeleton(car_pos, car_angle)) < 0.1:
        reward += 1

    if min(get_ray_distances_skeleton(car_pos, car_angle)) < 0.05:
        reward += 2

    # Small reward based on the car's speed to encourage faster movement
    if (car_speed / MAX_SPEED) > 0.7:
        reward += (car_speed / MAX_SPEED) * 2

    reward += (steps_alive/10)

    return reward

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
        if(distance/ray_length > 0.8):
            ray_distances.append(5)
        if(distance/ray_length > 0.6):
            ray_distances.append(4)
        if(distance/ray_length > 0.4):
            ray_distances.append(3)
        if(distance/ray_length > 0.2):
            ray_distances.append(2)
        else:
            ray_distances.append(1)

        ray_distances.append(distance)  # Normalize distance to [0, 1]

        # draw_ray(car_pos, distance, angle)
    return ray_distances


# ------------------------------------ Game ------------------------------------------

Q = load_q_table()

# Main Game Loop for Q-learning Training
for episode in range(1000):  # Train for a number of episodes
    car_pos = start_pos[:]
    car_angle = 20
    car_speed = 2
    total_reward = 0
    steps_alive = 0

    while True:
        steps_alive += 1
        # Handle events to keep Pygame responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        #draw screen
        screen.blit(track_image, (0, 0))

        # Check if episode is over (collision or timeout)
        if check_collision() or total_reward > 5000:
            print(f"Episode {episode + 1}: Total Reward: {total_reward}")
            break

        # Get current state
        state = get_state()
        
        # Choose an action based on current state
        action = choose_action(state)

        # Move car based on action
        move_car(action)
        
        # Compute reward and check for collisions
        try:
            reward = compute_reward(steps_alive)
        except:
            break

        total_reward += reward

        # Get next state
        next_state = get_state()

        # Store experience in replay buffer
        store_experience(state, action, reward, next_state)

        # Periodically update the Q-table from the replay buffer
        if len(replay_buffer) > batch_size:
            train_from_replay()

        # # Update Q-table
        # update_q_table(state, action, reward, next_state)

        #draw car
        rotated_car = pygame.transform.rotate(car_img, -car_angle)
        car_rect = rotated_car.get_rect(center=(car_pos[0], car_pos[1]))

        screen.blit(rotated_car, car_rect.topleft)

        
        pygame.display.flip()
        clock.tick(300)

save_q_table()