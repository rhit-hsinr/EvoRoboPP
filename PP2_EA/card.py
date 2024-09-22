import numpy as np
import matplotlib.pyplot as plt

class MGA_card_problem:

    def __init__(self, genesize, popsize, recomprob, mutationprob, tournaments):
        self.genesize = genesize
        self.popsize = popsize
        self.recomprob = recomprob
        self.mutationprob = mutationprob
        self.tournaments = tournaments
        self.pop = np.random.randint(2, size=(popsize, genesize))  # 0 or 1 for each card
        self.bestfit = []  # To track the best fitness over generations
        self.best_distribution = []  # To track the distribution of cards in best individuals
    
    def fitnessfunction(self, genotype):
        # Addition pile and multiplication pile
        addition_pile = [i + 1 for i in range(self.genesize) if genotype[i] == 0]
        multiplication_pile = [i + 1 for i in range(self.genesize) if genotype[i] == 1]
        
        # Sum and product of the respective piles
        sum_addition = sum(addition_pile)
        product_multiplication = np.prod(multiplication_pile) if multiplication_pile else 1
        
        # Calculate fitness based on deviation from target values
        sum_diff = abs(sum_addition - 36)
        product_diff = abs(product_multiplication - 360)
        
        # Minimize the combined difference (fitness is better when it's closer to zero)
        fitness = sum_diff + product_diff
        return -fitness
    
    def run(self):
        # Track the best fitness for each generation
        for t in range(self.tournaments):
            # Pick two individuals for tournament selection
            a, b = np.random.choice(np.arange(self.popsize), 2, replace=False)
            # Pick the winner based on fitness
            if self.fitnessfunction(self.pop[a]) > self.fitnessfunction(self.pop[b]):
                winner = a
                loser = b
            else:
                winner = b
                loser = a
            
            # Transfect winner to loser (crossover)
            for g in range(self.genesize):
                if np.random.random() < self.recomprob:
                    self.pop[loser][g] = self.pop[winner][g]
            
            # Mutate loser
            for g in range(self.genesize):
                if np.random.random() < self.mutationprob:
                    self.pop[loser][g] = 1 - self.pop[loser][g]  # Flip 0 to 1 or 1 to 0
            
            # Calculate and store the best fitness every generation
            if t % self.popsize == 0:
                fits = np.zeros(self.popsize)
                for i in range(self.popsize):
                    fits[i] = self.fitnessfunction(self.pop[i])
                
                best_fit = np.max(fits)
                self.bestfit.append(best_fit)
                
                # Track the best individual and its distribution
                best_index = np.argmax(fits)
                self.best_distribution.append(self.pop[best_index])
                
                print(f"Generation {t // self.popsize}: Best Fitness = {best_fit}")