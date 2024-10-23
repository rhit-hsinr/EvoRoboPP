import numpy as np
from sim import PredatorPreySimulation
import functions
import ea

# Parameters of the neural network
layers = [3,5,2]

# Parameters of the evolutionary algorithm
genesize = np.sum(np.multiply(layers[1:],layers[:-1])) + np.sum(layers[1:]) 
print("Number of parameters:",genesize)
popsize = 50 
recombProb = 0.8
mutatProb = 0.01
tournaments = 10000*popsize 

# simulation parameters
boundary_x = 100
boundary_y = 100
num_simulations = 1
num_steps = 50

# best_individual = functions.load_best_individual()

ga = ea.MGA(functions.calculate_fitness, genesize, popsize, recombProb, mutatProb, tournaments,
            num_steps, boundary_x, boundary_y)

# if best_individual is not None:
#     ga.pop[0] = best_individual

ga.run()
ga.showFitness()


# print("Best indices:", ga.bestind)
# print("Last best index:", ga.bestind[-1])

best_individual = ga.pop[int(ga.bestind[-1])]
functions.save_best_individual(best_individual)
best_fitness = ga.bestfit[-1]
worst_individual = ga.pop[int(ga.worstind[-1])]
print(f"Best Individual: {best_individual}")
print(f"Best Fitness: {best_fitness}")

functions.visualize_best_worst(best_individual, worst_individual, num_steps, boundary_x, boundary_y)

# sim = PredatorPreySimulation(boundary_x, boundary_y, num_simulations, num_steps)
# sim.simulate()
# sim.plot_simulation()