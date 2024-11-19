import numpy as np
import functions
import ea

# layers = [4,5,5,2]

size = 15  # Number of neurons in CTRNN
genesize = size * size + size + size  # Weights + Biases + Time Constants
print("Number of parameters:", genesize)


# genesize = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) 
# print("Number of parameters:",genesize)
popsize = 50 
recombProb = 0.8
mutatProb = 0.01
tournaments = 100*popsize 

num_simulations = 1

ga = ea.MGA(functions.fitnessfunction, genesize, popsize, recombProb,
            mutatProb, tournaments)

ga.run()
ga.showFitness()

best_individual = ga.pop[int(ga.bestind[-1])]
functions.save_best_individual(best_individual)
best_fitness = ga.bestfit[-1]
worst_fitness = ga.worstfit[-1]
worst_individual = ga.pop[int(ga.worstind[-1])]
print(f"Best Individual: {best_individual}")
print(f"Best Fitness: {best_fitness}")
print(f"Worst Individual: {worst_individual}")
print(f"Worst Fitness: {worst_fitness}")

functions.visualize_best_worst(best_individual, worst_individual)