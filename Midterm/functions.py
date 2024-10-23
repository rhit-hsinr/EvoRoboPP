import numpy as np
from predator_prey import Predator, Prey
from nn import PredatorNN
import matplotlib.pyplot as plt
import os

def calculate_fitness(predator_params, num_steps, boundary_x, boundary_y):

    nn = PredatorNN(units_per_layer=[3,5,2])
    nn.setParams(predator_params)
    prey = Prey(50, 50, 3)
    predator = Predator(90, 90, 3, sensor_range=100, nn=nn)

    for _ in range(num_steps):
        prey.move(boundary_x, boundary_y)
        predator.move_towards_prey_NN(prey.position, boundary_x, boundary_y)
    
    final_distance = np.linalg.norm(predator.position - prey.position)

    fitness = 1/(final_distance + 1e-6)
    return fitness

def calculate_fitness(predator_params, num_steps, boundary_x, boundary_y):

    nn = PredatorNN(units_per_layer=[3, 5, 2])
    nn.setParams(predator_params)
    prey = Prey(50, 50, 3)
    predator = Predator(90, 90, 3, sensor_range=100, nn=nn)

    total_angle_error = 0  # Track cumulative angle error over time

    for _ in range(num_steps):
        prey.move(boundary_x, boundary_y)
        predator.move_towards_prey_NN(prey.position, boundary_x, boundary_y)

        # Calculate the angle error between predator's current direction and the prey
        direction_to_prey = prey.position - predator.position
        desired_angle = np.arctan2(direction_to_prey[1], direction_to_prey[0])  # Ideal angle to prey
        angle_error = np.abs(np.arctan2(np.sin(predator.angle - desired_angle), np.cos(predator.angle - desired_angle)))

        total_angle_error += angle_error  # Accumulate angle error over time

    # Final distance to the prey
    final_distance = np.linalg.norm(predator.position - prey.position)

    # Fitness is a combination of distance and orientation towards prey
    fitness = (1 / (final_distance + 1e-6)) - (1 * total_angle_error / num_steps)  # Add angle error penalty

    return fitness


def visualize_best_worst(predator_params_best, predator_params_worst, num_steps, boundary_x, boundary_y):
    # Create instances of the neural network for best and worst individuals
    nn_best = PredatorNN(units_per_layer=[3, 5, 2])
    nn_best.setParams(predator_params_best)
    
    nn_worst = PredatorNN(units_per_layer=[3, 5, 2])
    nn_worst.setParams(predator_params_worst)
    
    # Create prey and boundary
    prey = Prey(50, 50, 3)
    
    # Initialize predators with their respective neural networks
    predator_best = Predator(10, 10, 3, sensor_range=100, nn=nn_best)
    predator_worst = Predator(10, 90, 3, sensor_range=100, nn=nn_worst)
    
    # Store positions for plotting
    best_positions = []
    worst_positions = []
    prey_positions = []

    for _ in range(num_steps):
        prey.move(boundary_x, boundary_y)
        predator_best.move_towards_prey_NN(prey.position, boundary_x, boundary_y)
        predator_worst.move_towards_prey_NN(prey.position, boundary_x, boundary_y)
        
        # Record positions
        best_positions.append(predator_best.position.copy())
        worst_positions.append(predator_worst.position.copy())
        prey_positions.append(prey.position.copy())

    best_positions = np.array(best_positions)
    worst_positions = np.array(worst_positions)
    prey_positions = np.array(prey_positions)

    # Plot the movements
    plt.figure(figsize=(10, 6))
    plt.plot(best_positions[:, 0], best_positions[:, 1], label="Best Predator", color="red")
    plt.plot(worst_positions[:, 0], worst_positions[:, 1], label="Worst Predator", color="blue")
    plt.plot(prey_positions[:, 0], prey_positions[:, 1], label="Prey", color="green")
    plt.scatter(prey_positions[-1, 0], prey_positions[-1, 1], color='green', s=50, marker='o', zorder=10, label='Prey Final')
    plt.scatter(best_positions[-1, 0], best_positions[-1, 1], color='red', s=50, marker='X', zorder=10, label='Best Predator Final')
    plt.scatter(worst_positions[-1, 0], worst_positions[-1, 1], color='blue', s=50, marker='X', zorder=10, label='Worst Predator Final')
    plt.xlim(0, boundary_x)
    plt.ylim(0, boundary_y)
    plt.title("Predator and Prey Movement: Best vs Worst")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid()
    plt.show()

def save_best_individual(best_individual, filename='best_individual.npy'):
    np.save(filename, best_individual)
    print(f"Best individual saved to {filename}")

import os

# Load the best individual from a .npy file if it exists
def load_best_individual(filename='best_individual.npy'):
    if os.path.exists(filename):
        best_individual = np.load(filename)
        print(f"Best individual loaded from {filename}")
        return best_individual
    else:
        print(f"No previous best individual found, starting fresh.")
        return None
