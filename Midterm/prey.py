import numpy as np
import matplotlib.pyplot as plt

class Prey:
    def __init__(self, x, y, speed):
        self.position = np.array([x,y], dtype=float)
        self.speed = speed
        self.angle = np.random.uniform(0, 2 * np.pi)
        self.numSteps = 0       #this keeps track of steps in certain direction
    
    def move(self, boundary_x, boundary_y):
        # Increase the range of random angle changes to make the prey more unpredictable
        random_angle = np.random.uniform(-np.pi / 8, np.pi / 8) 
        self.angle += random_angle

        direction_vector = np.array([np.cos(self.angle), np.sin(self.angle)])
        self.position += direction_vector * self.speed

        # Handle boundary collisions
        if self.position[0] < 0 or self.position[0] > boundary_x:
            self.position[0] = np.clip(self.position[0], 0, boundary_x)
            self.angle = np.pi - self.angle
        if self.position[1] < 0 or self.position[1] > boundary_y:
            self.position[1] = np.clip(self.position[1], 0, boundary_y)
            self.angle = -self.angle

        
