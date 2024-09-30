import numpy as np
import matplotlib.pyplot as plt
from neuron import NeuralNet

class GeneticAlgorithm:
    def __init__(self, population_size, hidden_units, weight_range, mutation_rate, generations):
        self.population_size = population_size
        self.hidden_units = hidden_units
        self.weight_range = weight_range
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = [NeuralNet(hidden_units, weight_range) for _ in range(population_size)]
        self.best_fitness_history = []

    def evaluate_fitness(self, neural_net):
        # Define the XOR inputs and corresponding outputs
        xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        xor_outputs = np.array([0, 1, 1, 0])

        # Calculate the total error for this neural network
        total_error = 0
        for i in range(len(xor_inputs)):
            output = neural_net.forward(xor_inputs[i])
            total_error += abs(output - xor_outputs[i])  # Mean squared error
        return -total_error  # Return negative error as fitness (maximize fitness)

    def select_parents(self):
        fitness_scores = [self.evaluate_fitness(nn) for nn in self.population]
        probabilities = fitness_scores / np.sum(fitness_scores)  # Normalize to get selection probabilities
        parents_indices = np.random.choice(range(self.population_size), size=2, p=probabilities)
        return parents_indices

    def crossover(self, parent1, parent2):
        # Create new child neural network
        child = NeuralNet(self.hidden_units, self.weight_range)

        # Get weights from both parents
        hidden_weights1, output_weights1 = parent1.get_weights()
        hidden_weights2, output_weights2 = parent2.get_weights()

        # Crossover the weights for hidden layer
        for i in range(self.hidden_units):
            if np.random.rand() < 0.5:
                hidden_weights1[i] = hidden_weights2[i]

        # Crossover for output layer weights
        if np.random.rand() < 0.5:
            output_weights1 = output_weights2

        child.set_weights(hidden_weights1, output_weights1)
        return child

    def mutate(self, neural_net):
        hidden_weights, output_weights = neural_net.get_weights()

        # Mutate hidden layer weights
        for i in range(len(hidden_weights)):
            for j in range(len(hidden_weights[i][0])):  # For each weight in hidden layer
                if np.random.rand() < self.mutation_rate:
                    hidden_weights[i][0][j] += np.random.uniform(-self.weight_range, self.weight_range)

        # Mutate output layer weights
        for j in range(len(output_weights[0])):  # For each weight in output layer
            if np.random.rand() < self.mutation_rate:
                output_weights[0][j] += np.random.uniform(-self.weight_range, self.weight_range)

        neural_net.set_weights(hidden_weights, output_weights)

    def run(self):
        for generation in range(self.generations):
            new_population = []

            # Generate new population through selection, crossover, and mutation
            for _ in range(self.population_size):
                parent_indices = self.select_parents()
                parent1 = self.population[parent_indices[0]]
                parent2 = self.population[parent_indices[1]]

                child = self.crossover(parent1, parent2)
                self.mutate(child)

                new_population.append(child)

            self.population = new_population  # Replace old population with new one

            # Optionally print the best fitness in each generation
            best_fitness = max(self.evaluate_fitness(nn) for nn in self.population)
            self.best_fitness_history.append(best_fitness)
            print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

        # Return the best neural network found
        best_network = max(self.population, key=self.evaluate_fitness)
        return best_network
    def viz(neuralnet, dataset, label):
        X = np.linspace(-1.05, 1.05, 100)
        Y = np.linspace(-1.05, 1.05, 100)
        output = np.zeros((100,100))
        i = 0
        for x in X: 
            j = 0
            for y in Y: 
                output[i,j] = neuralnet.forward([x,y])
                j += 1
            i += 1
        plt.contourf(X,Y,output)
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        for i in range(len(dataset)):
            if label[i] == 1:
                plt.plot(dataset[i][0],dataset[i][1],'wo')
            else:
                plt.plot(dataset[i][0],dataset[i][1],'wx')
        plt.show()  