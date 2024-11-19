import numpy as np
from predator_prey import Predator, Prey
from nn import FNN
from ctrnn import CTRNN
import matplotlib.pyplot as plt
import os

def fitnessfunction(genotype): # FNN
    # Initialize predator neural network and agent
    predatorNN = FNN([4, 5, 5, 2])  # Assuming FNN is the neural network class
    predatorNN.setParams(genotype)
    
    # Create prey and predator
    prey = Prey(0.5, 80, 80)
    predator = Predator(2, 50, 10, np.pi / 2, 1, predatorNN)
    
    # Params
    initial_distance = predator.distance(prey)
    max_steps = 200
    distance_threshold = 5
    
    # Tracking variables
    total_distance = 0
    min_distance = initial_distance
    steps_taken = max_steps
    previous_orientation = predator.orientation
    capture_reward = 1  # Reward if predator catches prey
    
    # Tracking the predator's performance
    for step in range(max_steps):
        predator.move(prey)
        prey.move(100, 100)
        # prey.move_rigid(max_steps, step)
        
        current_distance = predator.distance(prey)
        total_distance += current_distance
        min_distance = min(min_distance, current_distance)
        
        # Penalize erratic movement by comparing orientation changes
        orientation_diff = np.abs(predator.orientation - previous_orientation)
        orientation_penalty = np.exp(-orientation_diff)
        
        # # Update previous orientation for next step
        previous_orientation = predator.orientation
        
        # If predator catches prey, record the number of steps taken
        # if current_distance <= distance_threshold:
        #     steps_taken = step
        #     break

    # Fitness calculation
    final_distance = predator.distance(prey)
    
    # Average distance moved
    avg_distance = (total_distance - initial_distance) / steps_taken if steps_taken > 0 else 0

    # avg total distance
    avg_distance_factor = 1 / (1 + avg_distance)
    
    # final distance
    distance_factor = 1 / (1 + final_distance)
    
    # reward for capturing quickly
    time_factor = (max_steps - steps_taken) / max_steps if steps_taken < max_steps else 0
    
    # erratic movement
    # erratic_penalty = orientation_penalty
    
    capture_bonus = capture_reward if steps_taken < max_steps else 0
    
    fitness_score = (
        0.7 * avg_distance_factor +          # Reward for staying close to the prey (efficiency)
        0.3 * distance_factor        # Reward for approaching the prey
        # 0.1 * time_factor +           # Reward for capturing quickly
        # 0.1 * capture_bonus         # Extra reward for capturing prey
        # 0.1 * erratic_penalty         # Penalize erratic behavior (smooth movement)
    )
    
    return fitness_score

def fitnessfunction1(genotype): # CTRNN
    # Initialize predator CTRNN
    predatorNN = CTRNN(size=15)  # Adjust size based on your CTRNN design (e.g., 4 inputs, 9 hidden, 2 outputs)
    predatorNN.initializeState(np.zeros(predatorNN.Size))
    predatorNN.setParameters(
        weights=genotype[:15 * 15].reshape(15, 15),  # First part of genotype for weights
        biases=genotype[15 * 15:15 * 15 + 15],      # Next part for biases
        timeconstants=genotype[15 * 15 + 15:]       # Last part for time constants
    )
    
    # Create prey and predator
    prey = Prey(0.5, 80, 80)
    predator = Predator(2, 50, 10, np.pi / 2, 1, predatorNN)
    
    # Params
    initial_distance = predator.distance(prey)
    max_steps = 200
    distance_threshold = 5
    
    # Tracking variables
    total_distance = 0
    min_distance = initial_distance
    steps_taken = max_steps
    previous_orientation = predator.orientation
    capture_reward = 1  # Reward if predator catches prey
    
    # Tracking the predator's performance
    for step in range(max_steps):
        predator.move(prey)  # Uses the CTRNN in the predator's move function
        prey.move(100, 100)  # Moves prey randomly within the environment
        
        current_distance = predator.distance(prey)
        total_distance += current_distance
        min_distance = min(min_distance, current_distance)
        
        # Penalize erratic movement by comparing orientation changes
        orientation_diff = np.abs(predator.orientation - previous_orientation)
        orientation_penalty = np.exp(-orientation_diff)
        
        # Update previous orientation for next step
        previous_orientation = predator.orientation
        
        # If predator catches prey, record the number of steps taken
        if current_distance <= distance_threshold:
            steps_taken = step
            break

    # Fitness calculation
    final_distance = predator.distance(prey)
    
    # Average distance moved
    avg_distance = (total_distance - initial_distance) / steps_taken if steps_taken > 0 else 0

    # Average total distance factor
    avg_distance_factor = 1 / (1 + avg_distance)
    
    # Final distance factor
    distance_factor = 1 / (1 + final_distance)
    
    # Reward for capturing quickly
    time_factor = (max_steps - steps_taken) / max_steps if steps_taken < max_steps else 0
    
    # Capture bonus
    capture_bonus = capture_reward if steps_taken < max_steps else 0
    
    fitness_score = (
        0.7 * avg_distance_factor +  # Reward for staying close to the prey (efficiency)
        0.3 * distance_factor        # Reward for approaching the prey
        # 0.1 * time_factor +        # Reward for capturing quickly (optional)
        # 0.1 * capture_bonus        # Extra reward for capturing prey (optional)
    )
    
    return fitness_score



def visualize_best_worst(genotype_best, genotype_worst): # FNN
    # Create instances of the neural network for best and worst individuals
    nn_best = FNN([4, 5, 5, 2])
    nn_best.setParams(genotype_best)
    
    nn_worst = FNN([4, 5, 5, 2])
    nn_worst.setParams(genotype_worst)
    
    # Create prey and boundary
    prey_best = Prey(0.5,80,80)
    prey_worst = Prey(0.5,80,80)
    
    # Initialize predators with their respective neural networks
    predator_best = Predator(2,50,10,np.pi/2,1,nn_best)
    predator_worst = Predator(2,50,10,np.pi/2,1,nn_worst)
    
    # Store positions for plotting
    best_positions = []
    worst_positions = []
    prey_positions = []

     # params
    max_steps = 200
    distance_threshold = 5

    # tracking
    steps_taken_best = max_steps
    steps_taken_worst = max_steps

    for step in range(max_steps):
        predator_best.move(prey_best)
        predator_worst.move(prey_worst)
        prey_best.move(100,100)
        # prey_best.move_rigid(max_steps, step)

        current_distance_best = predator_best.distance(prey_best)
        current_distance_worst = predator_worst.distance(prey_worst)

        # if current_distance_worst <= distance_threshold:
        #     steps_taken_worst = step
        #     break

        best_positions.append((predator_best.xpos, predator_best.ypos))
        worst_positions.append((predator_worst.xpos, predator_worst.ypos))
        prey_positions.append(prey_best.position.copy())

    best_positions = np.array(best_positions)
    worst_positions = np.array(worst_positions)
    prey_positions = np.array(prey_positions)

    # Plot the movements
    plt.figure(figsize=(10, 6))
    plt.plot(best_positions[:, 0], best_positions[:, 1], label="Best Predator", color="red")
    plt.plot(worst_positions[:, 0], worst_positions[:, 1], label="Worst Predator", color="blue")
    plt.plot(prey_positions[:, 0], prey_positions[:, 1], label="Prey", color="green")

    plt.scatter(best_positions[-1, 0], best_positions[-1, 1], color='red', s=50, marker='X', zorder=10, label='Best Predator Final')
    plt.scatter(worst_positions[-1, 0], worst_positions[-1, 1], color='blue', s=50, marker='X', zorder=10, label='Worst Predator Final')
    plt.scatter(prey_positions[-1, 0], prey_positions[-1, 1], color='green', s=50, marker='X', zorder=10, label='Prey Final')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.title("Predator and Prey Movement: Best vs Worst")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid()
    plt.show()

def visualize_best_worst1(genotype_best, genotype_worst): # CTRNN
    # Define CTRNN structure
    num_neurons = 15
    num_weights = num_neurons * num_neurons  # Fully connected weights
    num_biases = num_neurons
    num_timeconstants = num_neurons

    def decode_genotype(genotype):
        """Decode genotype into weights, biases, and time constants."""
        weights = genotype[:num_weights].reshape((num_neurons, num_neurons))
        biases = genotype[num_weights:num_weights + num_biases]
        timeconstants = genotype[num_weights + num_biases:]
        return weights, biases, timeconstants

    # Decode and set parameters for best and worst CTRNNs
    ctrnn_best = CTRNN(size=num_neurons)
    weights_best, biases_best, timeconstants_best = decode_genotype(genotype_best)
    ctrnn_best.setParameters(weights_best, biases_best, timeconstants_best)
    ctrnn_best.initializeState(np.zeros(num_neurons))  # Initialize state

    ctrnn_worst = CTRNN(size=num_neurons)
    weights_worst, biases_worst, timeconstants_worst = decode_genotype(genotype_worst)
    ctrnn_worst.setParameters(weights_worst, biases_worst, timeconstants_worst)
    ctrnn_worst.initializeState(np.zeros(num_neurons))  # Initialize state

    # Create prey
    prey_best = Prey(0.5, 80, 80)
    prey_worst = Prey(0.5, 80, 80)

    # Initialize predators with their respective CTRNNs
    predator_best = Predator(2, 50, 10, np.pi / 2, 1, ctrnn_best)
    predator_worst = Predator(2, 50, 10, np.pi / 2, 1, ctrnn_worst)

    # Store positions for plotting
    best_positions = []
    worst_positions = []
    prey_positions = []

    # Simulation parameters
    max_steps = 200
    distance_threshold = 5

    # Track steps
    for step in range(max_steps):
        predator_best.move(prey_best)
        predator_worst.move(prey_worst)
        prey_best.move(100, 100)

        best_positions.append((predator_best.xpos, predator_best.ypos))
        worst_positions.append((predator_worst.xpos, predator_worst.ypos))
        prey_positions.append(prey_best.position.copy())

    best_positions = np.array(best_positions)
    worst_positions = np.array(worst_positions)
    prey_positions = np.array(prey_positions)

    # Plot the movements
    plt.figure(figsize=(10, 6))
    plt.plot(best_positions[:, 0], best_positions[:, 1], label="Best Predator", color="red")
    plt.plot(worst_positions[:, 0], worst_positions[:, 1], label="Worst Predator", color="blue")
    plt.plot(prey_positions[:, 0], prey_positions[:, 1], label="Prey", color="green")

    # Mark final positions
    plt.scatter(best_positions[-1, 0], best_positions[-1, 1], color='red', s=50, marker='X', zorder=10, label='Best Predator Final')
    plt.scatter(worst_positions[-1, 0], worst_positions[-1, 1], color='blue', s=50, marker='X', zorder=10, label='Worst Predator Final')
    plt.scatter(prey_positions[-1, 0], prey_positions[-1, 1], color='green', s=50, marker='X', zorder=10, label='Prey Final')

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.title("Predator and Prey Movement: Best vs Worst (CTRNN)")
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
