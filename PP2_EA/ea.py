import numpy as np

class MGA_disc():
    def __init__(self, fitnessfunction, genesize, popsize, recomprob, mutationprob, tournaments):
        self.genesize = genesize
        self.popsize = popsize
        self.recomprob = recomprob
        self.mutationprob = mutationprob
        self.tournaments = tournaments
        self.fitnessfunction = fitnessfunction
        self.pop = np.random.randint(2, size=(popsize, genesize))
        self.fitness = np.apply_along_axis(self.fitnessfunction, 1, self.pop)  # Calculate fitness initially

    def run(self):
        # 1 loop for tour
        for t in range(self.tournaments):
            # 2 pick two to fight without replacement
            a, b = np.random.choice(self.popsize, 2, replace=False)
            
            # 3 pick winner based on fitness
            if self.fitness[a] > self.fitness[b]:
                winner, loser = a, b
            else:
                winner, loser = b, a
            
            # 4 transfect winner to loser using vectorized approach
            mask = np.random.random(self.genesize) < self.recomprob
            self.pop[loser] = np.where(mask, self.pop[winner], self.pop[loser])
            
            # 5 mutate loser using vectorized mutation
            mutation_mask = np.random.random(self.genesize) < self.mutationprob
            self.pop[loser] = np.where(mutation_mask, 1 - self.pop[loser], self.pop[loser])
            
            # Update fitness for the loser only
            self.fitness[loser] = self.fitnessfunction(self.pop[loser])
            
            # 6 Stats output
            if t % self.popsize == 0:
                max_fit = np.max(self.fitness)
                mean_fit = np.mean(self.fitness)
                min_fit = np.min(self.fitness)
                best_individual = np.argmax(self.fitness)
                print(f"Iteration {t}: Max = {max_fit}, Mean = {mean_fit}, Min = {min_fit}, Best = {best_individual}")

class MGA_real():
    def __init__(self, fitnessfunction, genesize, popsize, recomprob, mutationprob, tournaments):
        self.genesize = genesize
        self.popsize = popsize
        self.recomprob = recomprob
        self.mutationprob = mutationprob
        self.tournaments = tournaments
        self.fitnessfunction = fitnessfunction
        self.pop = np.random.random((popsize, genesize))
        self.fitness = self.calculateFitness()  # Pre-calculate fitness

        # Stats
        gens = tournaments // popsize
        self.bestfit = np.zeros(gens)

    def calculateFitness(self):
        return np.apply_along_axis(self.fitnessfunction, 1, self.pop)

    def run(self):
        gen = 0
        for t in range(self.tournaments):
            # 2 pick two distinct individuals
            a, b = np.random.choice(self.popsize, 2, replace=False)

            # 3 pick the winner
            if self.fitness[a] > self.fitness[b]:
                winner, loser = a, b
            else:
                winner, loser = b, a
            
            # 4 recombine using vectorized operation
            recombination_mask = np.random.random(self.genesize) < self.recomprob
            self.pop[loser] = np.where(recombination_mask, self.pop[winner], self.pop[loser])
            
            # 5 mutate loser using vectorized normal distribution
            mutation_vector = np.random.normal(0, self.mutationprob, self.genesize)
            self.pop[loser] = np.clip(self.pop[loser] + mutation_vector, 0, 1)
            
            # Update fitness for the loser
            self.fitness[loser] = self.fitnessfunction(self.pop[loser])
            
            # 6 Stats
            if t % self.popsize == 0:
                self.bestfit[gen] = np.max(self.fitness)
                gen += 1
                print(f"Iteration {t}: Max = {np.max(self.fitness)}, Mean = {np.mean(self.fitness)}, Min = {np.min(self.fitness)}")



            
