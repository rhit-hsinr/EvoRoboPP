import numpy as np
from nn import FNN

class Predator:
    def __init__(self, speed, pred_xpos, pred_ypos, orientation, radius, nn):
        self.position = np.array([pred_xpos, pred_ypos], dtype=float)
        self.speed = speed
        self.xpos = pred_xpos
        self.ypos = pred_ypos
        self.orientation = orientation
        self.radius = radius
        self.leftsensor = 0
        self.rightsensor = 0
        self.leftmotor = 0
        self.rightmotor = 0
        self.previous_distance = None

        self.angleoffset = np.pi/4
        self.rs_xpos = self.radius * np.cos(self.orientation + self.angleoffset)
        self.rs_ypos = self.radius * np.sin(self.orientation + self.angleoffset)
        self.ls_xpos = self.radius * np.cos(self.orientation - self.angleoffset)
        self.ls_ypos = self.radius * np.sin(self.orientation - self.angleoffset)

        self.nn = nn

    def sense(self, prey):
        self.leftsensor = 1 - np.sqrt((self.ls_xpos-prey.xpos)**2 + (self.ls_ypos-prey.ypos)**2)/10
        self.leftsensor = np.clip(self.leftsensor,0,1)
        self.rightsensor = 1 - np.sqrt((self.rs_xpos-prey.ypos)**2 + (self.rs_ypos-prey.ypos)**2)/10
        self.rightsensor = np.clip(self.rightsensor,0,1)

        return self.leftsensor, self.rightsensor

    def move(self, prey): # FNN move
        leftsensor, rightsensor = self.sense(prey)
        relative_angle = self.relative_angle(prey)
        current_distance = self.distance(prey)
        distance_change = 0 if self.previous_distance is None else (self.previous_distance - current_distance)
        self.previous_distance = current_distance

        output = self.nn.forward([leftsensor, rightsensor, distance_change, relative_angle])
        self.rightmotor, self.leftmotor = output[0]

        self.rightmotor = np.clip(self.rightmotor,0,1)
        self.leftmotor  = np.clip(self.leftmotor,0,1)
        self.orientation += ((self.leftmotor - self.rightmotor)/2) + np.random.normal(0,0.1)
        self.speed = ((self.rightmotor + self.leftmotor)/2)
        
        # Update position of the agent
        self.xpos += self.speed * np.cos(self.orientation) 
        self.ypos += self.speed * np.sin(self.orientation)  
        
        # Update position of the sensors
        self.rs_xpos = self.xpos + self.radius * np.cos(self.orientation + self.angleoffset)
        self.rs_ypos = self.ypos + self.radius * np.sin(self.orientation + self.angleoffset)
        self.ls_xpos = self.xpos + self.radius * np.cos(self.orientation - self.angleoffset)
        self.ls_ypos = self.ypos + self.radius * np.sin(self.orientation - self.angleoffset)

    def move1(self, prey): # CTRNN move
        leftsensor, rightsensor = self.sense(prey)
        relative_angle = self.relative_angle(prey)
        current_distance = self.distance(prey)
        distance_change = 0 if self.previous_distance is None else (self.previous_distance - current_distance)
        self.previous_distance = current_distance

        # Set CTRNN inputs
        self.nn.Inputs[:4] = [leftsensor, rightsensor, distance_change, relative_angle]  # First 4 neurons are inputs

        # Step the CTRNN
        dt = 0.1  # Define the time step
        self.nn.step(dt)

        # Get outputs from CTRNN (use specific neurons for output)
        self.rightmotor = np.clip(self.nn.Outputs[-2], 0, 1)  # Second-to-last neuron is the right motor
        self.leftmotor = np.clip(self.nn.Outputs[-1], 0, 1)   # Last neuron is the left motor

        # Update orientation and speed
        self.orientation += ((self.leftmotor - self.rightmotor) / 2) + np.random.normal(0, 0.1)
        self.speed = (self.rightmotor + self.leftmotor) / 2

        # Update position of the agent
        self.xpos += self.speed * np.cos(self.orientation)
        self.ypos += self.speed * np.sin(self.orientation)

        # Update position of the sensors
        self.rs_xpos = self.xpos + self.radius * np.cos(self.orientation + self.angleoffset)
        self.rs_ypos = self.ypos + self.radius * np.sin(self.orientation + self.angleoffset)
        self.ls_xpos = self.xpos + self.radius * np.cos(self.orientation - self.angleoffset)
        self.ls_ypos = self.ypos + self.radius * np.sin(self.orientation - self.angleoffset)

    def distance(self, prey):
        # return np.sqrt((self.xpos-prey.xpos)**2 + (self.ypos-prey.ypos)**2)
        return np.sqrt((self.xpos-prey.position[0])**2 + (self.ypos-prey.position[1])**2)
    # def distance(self, prey):
    #     # Calculate Euclidean distance between predator's position and prey's position
    #     return np.linalg.norm(self.position - prey.position)

    def relative_angle(predator, prey):
        # Vector from predator to prey
        vector_to_prey = np.array([prey.position[0] - predator.position[0], 
                                prey.position[1] - predator.position[1]])
        
        # Predator's orientation vector based on its angle
        predator_orientation_vector = np.array([np.cos(predator.orientation), np.sin(predator.orientation)])
        
        # Calculate the angle between predator's orientation and the prey
        dot_product = np.dot(predator_orientation_vector, vector_to_prey)
        magnitude_product = np.linalg.norm(predator_orientation_vector) * np.linalg.norm(vector_to_prey)
        
        # Calculate the angle in radians
        angle = np.arccos(dot_product / magnitude_product)
        
        # Determine the direction (left or right) by the cross product
        cross_product = predator_orientation_vector[0] * vector_to_prey[1] - predator_orientation_vector[1] * vector_to_prey[0]
        if cross_product < 0:
            angle = -angle  # Angle is negative if prey is to the left of the predator's orientation
        
        return angle



class Prey:
    def __init__(self, speed, xpos, ypos):
        self.position = np.array([xpos, ypos], dtype=float)
        self.xpos = xpos
        self.ypos = ypos
        self.speed = speed
        self.angle = np.pi/2
        self.numSteps = 0
        self.is_alive = False
    
    def move(self, boundary_x, boundary_y):
        if (self.is_alive):
            random_angle = np.random.uniform(-np.pi / 8, np.pi / 8) 
            self.angle += random_angle

            direction_vector = np.array([np.cos(self.angle), np.sin(self.angle)])
            self.position += direction_vector * self.speed

            if self.position[0] < 0 or self.position[0] > boundary_x:
                self.position[0] = np.clip(self.position[0], 0, boundary_x)
                self.angle = np.pi - self.angle
            if self.position[1] < 0 or self.position[1] > boundary_y:
                self.position[1] = np.clip(self.position[1], 0, boundary_y)
                self.angle = -self.angle
        else:
            pass

    def move_rigid(self, maxSteps, numSteps):
        if (self.is_alive):
            quad = maxSteps / 4
            if (numSteps % quad != 0):
                self.position += self.speed
            else:
                self.angle += np.pi/2
        else:
            pass
            