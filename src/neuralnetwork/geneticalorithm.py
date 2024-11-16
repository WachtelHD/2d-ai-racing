import torch
import torch.nn as nn
import numpy as np
import random
import copy

# Neural network model for each agent --------------> exists in neural network file
class CarAIModel(nn.Module):
    def __init__(self, input_size):
        super(CarAIModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Outputs: throttle and steering

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

# Parameters
population_size = 50
generations = 100
mutation_rate = 0.1
num_top_agents = 10

# Initialize a population of models
def initialize_population(input_size):
    return [CarAIModel(input_size) for _ in range(population_size)]

# Evaluate each agent by running it through the environment
def evaluate_agent(agent):
    # Simulate environment interactions here and calculate fitness
    # For example: sum of distances traveled without collision
    return random.uniform(0, 100)  # Placeholder for actual fitness function

# Perform crossover between two parent agents
def crossover(parent1, parent2):
    child = copy.deepcopy(parent1)
    for param_child, param_parent1, param_parent2 in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
        param_child.data.copy_((param_parent1.data + param_parent2.data) / 2)  # Average weights
    return child

# Mutate the model slightly by adding noise
def mutate(agent, mutation_rate=0.1):
    for param in agent.parameters():
        if random.random() < mutation_rate:
            param.data += torch.randn_like(param) * 0.1  # Small random change
    return agent

# Main genetic algorithm loop
def train_population(input_size):
    population = initialize_population(input_size)
    
    for generation in range(generations):
        print(f"Generation {generation+1}")
        
        # Evaluate fitness of each agent
        fitness_scores = [(agent, evaluate_agent(agent)) for agent in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by fitness score
        
        # Select the top agents
        top_agents = [agent for agent, score in fitness_scores[:num_top_agents]]
        print(f"Top score: {fitness_scores[0][1]}")
        
        # Create new population
        new_population = []
        while len(new_population) < population_size:
            # Randomly select two parents from top agents
            parent1, parent2 = random.sample(top_agents, 2)
            child = crossover(parent1, parent2)  # Create child through crossover
            child = mutate(child, mutation_rate)  # Apply mutation
            new_population.append(child)
        
        # Replace old population with the new one
        population = new_population

# Run the training
input_size = 10  # For example, 8 ray distances + speed + angle
train_population(input_size)