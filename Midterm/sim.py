import matplotlib.pyplot as plt
from predator_prey import Predator, Prey
from nn import PredatorNN
import numpy as np

class PredatorPreySimulation:
    def __init__(self, boundary_x, boundary_y, num_simulations, num_steps):
        self.boundary_x = boundary_x
        self.boundary_y = boundary_y
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.all_prey_paths = []
        self.all_predator_paths = []

    def simulate(self): # Change the parameters here for simulation
        for sim in range(self.num_simulations):

            # Initialization of Prey, Predator, NN
            predator_nn = PredatorNN([3, 5, 2], activations='sigmoid')
            nn_params = np.random.uniform(-1, 1, predator_nn.get_num_params())  # Randomize NN parameters for now
            predator_nn.setParams(nn_params)
            prey = Prey(50, 50, 3)
            predator = Predator(10, 10, 3, sensor_range=50, nn=predator_nn)
            

            prey_path = [prey.position.copy()]
            predator_path = [predator.position.copy()]

            for _ in range(self.num_steps):
                prey.move(self.boundary_x, self.boundary_y)
                predator.move_towards_prey_NN(prey.position, self.boundary_x, self.boundary_y)

                prey_path.append(prey.position.copy())
                predator_path.append(predator.position.copy())

            self.all_prey_paths.append(prey_path)
            self.all_predator_paths.append(predator_path)

    def plot_simulation(self):
        plt.figure(figsize=(7, 5))

        # Track if the legend has been added for each element
        prey_path_label_added = False
        predator_path_label_added = False

        # Plot each simulation path
        for i in range(self.num_simulations):
            prey_path = np.array(self.all_prey_paths[i])
            predator_path = np.array(self.all_predator_paths[i])

            # Debug: Print final positions to verify they are being calculated correctly
            print(f"Prey final position (Simulation {i+1}): {prey_path[-1]}")
            print(f"Predator final position (Simulation {i+1}): {predator_path[-1]}")

            # Plot paths for prey and predator
            plt.plot(prey_path[:, 0], prey_path[:, 1], color='blue', linestyle='--', label='Prey Path' if not prey_path_label_added else "")
            plt.plot(predator_path[:, 0], predator_path[:, 1], color='red', label='Predator Path' if not predator_path_label_added else "")

            # Add a dot for the final positions
            plt.scatter(prey_path[-1, 0], prey_path[-1, 1], color='blue', s=50, marker='o', zorder=10, label='Prey Final' if not prey_path_label_added else "")
            plt.scatter(predator_path[-1, 0], predator_path[-1, 1], color='red', s=50, marker='X', zorder=10, label='Predator Final' if not predator_path_label_added else "")

            # Update flags to only show labels once
            prey_path_label_added = True
            predator_path_label_added = True

        # Set plot boundaries
        plt.xlim(0, self.boundary_x)
        plt.ylim(0, self.boundary_y)

        # Labels and grid
        plt.title(f"Predator-Prey Simulation ({self.num_simulations} runs)")
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()