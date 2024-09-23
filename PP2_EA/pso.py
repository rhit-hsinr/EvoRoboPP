import numpy as np
import matplotlib.pyplot as plt

def quadratic_function(x):
    return x**2

# Particle class
class Particle:
    def __init__(self, lower_bound, upper_bound):
        self.position = np.random.uniform(low=lower_bound, high=upper_bound)
        self.velocity = np.random.uniform(low=-1, high=1)
        
        self.personal_best_position = self.position
        self.personal_best_score = quadratic_function(self.position)

    # Update the particle's velocity and position
    def update_velocity(self, w, c1, c2, global_best_position):
        r1 = np.random.rand()
        r2 = np.random.rand()
        
        cognitive_component = c1 * r1 * (self.personal_best_position - self.position)
        social_component = c2 * r2 * (global_best_position - self.position)
        
        self.velocity = w * self.velocity + cognitive_component + social_component

    # Update the position of the particle
    def update_position(self, lower_bound, upper_bound):
        self.position += self.velocity
        self.position = np.clip(self.position, lower_bound, upper_bound)  # Ensure position is within bounds

    # Evaluate the current position and update the personal best if necessary
    def evaluate(self):
        fitness = quadratic_function(self.position)
        if fitness < self.personal_best_score:
            self.personal_best_score = fitness
            self.personal_best_position = self.position

# PSO class
class PSO:
    def __init__(self, n_particles, n_iterations, w, c1, c2, lower_bound, upper_bound):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive component
        self.c2 = c2  # Social component
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        # Initialize particles
        self.particles = [Particle(lower_bound, upper_bound) for _ in range(n_particles)]
        
        # Initialize global best
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.best_scores = []

    # Run the PSO optimization process
    def optimize(self):
        for iteration in range(self.n_iterations):
            for particle in self.particles:
                # Evaluate each particle and update its personal best
                particle.evaluate()
                
                # Update global best if necessary
                if particle.personal_best_score < self.global_best_score:
                    self.global_best_score = particle.personal_best_score
                    self.global_best_position = particle.personal_best_position

            # Update velocity and position for each particle
            for particle in self.particles:
                particle.update_velocity(self.w, self.c1, self.c2, self.global_best_position)
                particle.update_position(self.lower_bound, self.upper_bound)

            # Record the best score of this iteration
            self.best_scores.append(self.global_best_score)

            # Print progress
            print(f"Iteration {iteration+1}/{self.n_iterations}, Best score: {self.global_best_score}")


