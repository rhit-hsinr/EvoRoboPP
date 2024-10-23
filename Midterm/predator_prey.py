import numpy as np
import matplotlib.pyplot as plt
from nn import PredatorNN

class Predator:
    def __init__(self, x, y, speed, sensor_range, nn):
        self.position = np.array([x, y], dtype=float)
        self.speed = speed
        self.angle = np.random.uniform(0, 2 * np.pi)
        self.sensor_range = sensor_range
        self.nn = nn

    def get_sensor_data(self, prey_position):
        direction_to_prey = prey_position - self.position
        distance_to_prey = np.linalg.norm(direction_to_prey)

        # Angle between predator's current movement direction and the prey
        angle_to_prey = np.arctan2(direction_to_prey[1], direction_to_prey[0]) - self.angle
        angle_to_prey = np.arctan2(np.sin(angle_to_prey), np.cos(angle_to_prey))  # Normalize angle to [-pi, pi]

        # Sensor activation: the closer and more aligned the prey, the higher the sensor data
        sensor_left = 0
        sensor_right = 0
        if distance_to_prey <= self.sensor_range:
            # Prey is on the right side (negative angles)
            if -np.pi / 4 <= angle_to_prey <= 0:  
                sensor_right = 1 - abs(angle_to_prey / (np.pi / 4))  # Right sensor activation
            # Prey is on the left side (positive angles)
            elif 0 <= angle_to_prey <= np.pi / 4: 
                sensor_left = 1 - abs(angle_to_prey / (np.pi / 4))  # Left sensor activation

        return sensor_left, sensor_right, distance_to_prey
    
    
    # ------------------------------------------------------------
    # Pre-movement without using Neural Network
    # ------------------------------------------------------------
    def move_towards_prey(self, prey_position, boundary_x, boundary_y):

        # Calculate the direction to the prey
        direction_to_prey = prey_position - self.position
        distance_to_prey = np.linalg.norm(direction_to_prey)
        
        if distance_to_prey > 0:  # Avoid division by zero
            # Normalize the direction vector to get a unit vector
            direction_to_prey /= distance_to_prey

        # Calculate the angle between predator's current direction and the prey
        desired_angle = np.arctan2(direction_to_prey[1], direction_to_prey[0])

        # Rotate the predator smoothly towards the prey by updating its angle
        angle_diff = desired_angle - self.angle
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))  # Normalize the angle

        # Adjust angle for moving towards prey
        max_turn_angle = np.pi / 16 
        if angle_diff > max_turn_angle:
            self.angle += max_turn_angle
        elif angle_diff < -max_turn_angle:
            self.angle -= max_turn_angle
        else:
            self.angle = desired_angle  # If the angle difference is small, align exactly

        # Update the position of the predator based on the updated angle
        direction_vector = np.array([np.cos(self.angle), np.sin(self.angle)])
        self.position += direction_vector * self.speed

        # Handle boundary collisions
        if self.position[0] < 0 or self.position[0] > boundary_x:
            self.angle = np.pi - self.angle
        if self.position[1] < 0 or self.position[1] > boundary_y:
            self.angle = -self.angle

    # ------------------------------------------------------------
    # Movement using Neural Network
    # ------------------------------------------------------------
    def move_towards_prey_NN(self, prey_position, boundary_x, boundary_y):
        # Get sensor data
        sensor_left, sensor_right, normalized_distance = self.get_sensor_data(prey_position)

        # Input to the neural network: [sensor_left, sensor_right, normalized_distance]
        inputs = np.array([sensor_left, sensor_right, normalized_distance])

        # Forward pass through the neural network
        output = self.nn.forward(inputs)

        # Ensure output is correctly shaped
        if output.shape == (1, 2):
            angle_adjustment, speed_adjustment = output[0]  # Unpack the values from the 2D array
        else:
            raise ValueError("Weird {}".format(output.shape))
        
        # print(f"Speed: {speed_adjustment}, Angle: {angle_adjustment}")

        # Adjust the predator's angle and speed based on NN output
        max_turn_angle = np.pi / 4  # Limit how much the predator can turn in one step
        self.angle += max_turn_angle * np.clip(angle_adjustment, -1, 1)  # NN output controls angle
        # self.speed = np.clip(speed_adjustment + self.speed, 1, 2 * self.speed)  # Adjust speed
        self.speed = np.clip(self.speed * (1 + 0.5 * speed_adjustment), 1, 5)


        # Update position
        direction_vector = np.array([np.cos(self.angle), np.sin(self.angle)])
        self.position += direction_vector * self.speed

        # Boundary conditions
        if self.position[0] < 0:
            self.position[0] = 0
            self.angle = np.pi - self.angle
        elif self.position[0] > boundary_x:
            self.position[0] = boundary_x
            self.angle = np.pi - self.angle

        if self.position[1] < 0:
            self.position[1] = 0
            self.angle = -self.angle
        elif self.position[1] > boundary_y:
            self.position[1] = boundary_y
            self.angle = -self.angle


        # Debugging information
        # print("Predator position:", self.position)
        # print("Predator angle:", self.angle)
        # print("Predator speed:", self.speed)


#
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
#

class Prey:
    def __init__(self, x, y, speed):
        self.position = np.array([x,y], dtype=float)
        self.speed = speed
        self.angle = np.random.uniform(0, 2 * np.pi)
        self.numSteps = 0       #this keeps track of steps in certain direction
    
    def move(self, boundary_x, boundary_y):
        # Randomize the movement angle slightly to make the prey's movement less predictable
        random_angle = np.random.uniform(-np.pi / 8, np.pi / 8)
        self.angle += random_angle

        # Move the prey in the direction of the current angle
        direction_vector = np.array([np.cos(self.angle), np.sin(self.angle)])
        self.position += direction_vector * self.speed

        # Handle boundary conditions by reflecting position and adjusting angle
        if self.position[0] < 0:
            self.position[0] = 0
            self.angle = np.pi - self.angle
        elif self.position[0] > boundary_x:
            self.position[0] = boundary_x
            self.angle = np.pi - self.angle

        if self.position[1] < 0:
            self.position[1] = 0
            self.angle = -self.angle
        elif self.position[1] > boundary_y:
            self.position[1] = boundary_y
            self.angle = -self.angle

#
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
#