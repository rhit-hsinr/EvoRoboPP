import numpy as np
import ea 
import card
import matplotlib.pyplot as plt 

def fitnessfunction(genotype):
    return np.sum(genotype)

# Global variables
genesize = 10
popsize = 100
recomprob = 0.5
mutationprob = 1
tournaments = 10000

# g = ea.MGA_disc(fitnessfunction, genesize, popsize, recomprob, mutationprob, tournaments)
# print("\n Example evolutionary run using discrete genotype:")
# g.run()

# g = ea.MGA_real(fitnessfunction, genesize, popsize, recomprob, mutationprob, tournaments)
# print("\n Example evolutionary run using real-valued genotype:")
# g.run()

g = card.MGA_card_problem(genesize, popsize, recomprob, mutationprob, tournaments)

# Run the genetic algorithm
g.run()

# Plot the best fitness over generations
plt.plot(g.bestfit)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Best Fitness Over Generations')
plt.show()

# Analyze the last best distribution
final_best = g.best_distribution[-1]

# Output the final distribution of cards
addition_pile = [i + 1 for i in range(genesize) if final_best[i] == 0]
multiplication_pile = [i + 1 for i in range(genesize) if final_best[i] == 1]

print("\nFinal Distribution of Cards:")
print("Genotype Representation (0 = Add, 1 = Multiply):", final_best)
print("Addition Pile (Sum to get close to 36):", addition_pile)
print("Multiplication Pile (Product to get close to 360):", multiplication_pile)



