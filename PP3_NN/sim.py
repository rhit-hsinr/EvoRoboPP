import neuron
from ea import GeneticAlgorithm
import numpy as np
import matplotlib.pyplot as plt

reps = 1
trials = 10000

# A simple problem
data = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
label = np.array([0,1,1,0])
layers = [3,5,1]
error = np.zeros(trials)
a = neuron.NeuralNet(6,1)
a.viz2(100,"Before training",data,label)
for t in range(trials):
    for d in range(len(data)):
        error[t] += a.train(data[d],label[d])
a.viz2(100,"After training",data,label)

plt.plot(error.T)
plt.xlabel("Trials")
plt.ylabel("Error")
plt.show()

# population_size = 50
# hidden_units = 2
# weight_range = 1.0
# mutation_rate = 0.1
# generations = 100

# ga = GeneticAlgorithm(population_size, hidden_units, weight_range, mutation_rate, generations)
# best_nn = ga.run()

# # Plotting the fitness history
# plt.plot(ga.best_fitness_history)
# plt.xlabel('Generation')
# plt.ylabel('Best Fitness')
# plt.title('Fitness Over Generations')
# plt.grid()
# plt.show()

# # Visualize decision boundary using the best neural network
# best_nn.viz2(100, "Best Neural Network Decision Boundary", data, label)