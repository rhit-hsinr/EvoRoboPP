import numpy as np

class Vehicle:

    def __init__(self):
        
        self.xpos = 0.0                                       # agent's x position, starts in middle of world
        self.ypos = 0.0                                       # agent's y position, starts in middle of world
        self.orientation = 0.0 #np.random.random()*2*np.pi         # agent's orientation, starts at random
        self.velocity = 0.0                                   # agent's velocity, starts at 0
        self.radius = 1.0                                     # the size/radius of the vehicle
        self.leftsensor = 0.0                                 # left sensor value
        self.rightsensor = 0.0                                # right sensor value
        self.leftmotor  = 1.0                                 # left motor output
        self.rightmotor = 1.0                                 # right motor output
        
        # Attributes to determine the placement of the sensors
        self.angleoffset = np.pi/2                                                 # left/right sensor angle offset
        self.rs_xpos = self.radius * np.cos(self.orientation + self.angleoffset)   # right sensor x position
        self.rs_ypos = self.radius * np.sin(self.orientation + self.angleoffset)   # right sensor y position
        self.ls_xpos = self.radius * np.cos(self.orientation - self.angleoffset)   # left sensor x position
        self.ls_ypos = self.radius * np.sin(self.orientation - self.angleoffset)   # left sensor y position
        
    def sense(self,light):
        # Calculate the distance of the light for each of the sensors
        self.leftsensor = 1 - np.sqrt((self.ls_xpos-light.xpos)**2 + (self.ls_ypos-light.ypos)**2)/10
        self.leftsensor = np.clip(self.leftsensor,0,1)
        self.rightsensor = 1 - np.sqrt((self.rs_xpos-light.xpos)**2 + (self.rs_ypos-light.ypos)**2)/10
        self.rightsensor = np.clip(self.rightsensor,0,1)

    def think(self):
        ## Delete the pass command and add your code in this method
        ## In particular, think about activating the right and left motors 
        ## in the way that would make the vehicle move towards the light source
        ## (but also, feel free to change anything else in the code)
        self.rightmotor = self.leftsensor
        self.leftmotor = self.rightsensor

    def move(self):
        # Update the orientation and velocity of the vehicle based on the left and right motors
        self.rightmotor = np.clip(self.rightmotor,0,1)
        self.leftmotor  = np.clip(self.leftmotor,0,1)
        self.orientation += ((self.leftmotor - self.rightmotor)/10) + np.random.normal(0,0.1)
        self.velocity = ((self.rightmotor + self.leftmotor)/2)/50
        
        # Update position of the agent
        self.xpos += self.velocity * np.cos(self.orientation) 
        self.ypos += self.velocity * np.sin(self.orientation)  
        
        # Update position of the sensors
        self.rs_xpos = self.xpos + self.radius * np.cos(self.orientation + self.angleoffset)
        self.rs_ypos = self.ypos + self.radius * np.sin(self.orientation + self.angleoffset)
        self.ls_xpos = self.xpos + self.radius * np.cos(self.orientation - self.angleoffset)
        self.ls_ypos = self.ypos + self.radius * np.sin(self.orientation - self.angleoffset)

    def distance(self,light):
        return np.sqrt((self.xpos-light.xpos)**2 + (self.ypos-light.ypos)**2)

class Light:  

    def __init__(self):
        angle = 0.0 #np.random.random()*2*np.pi
        self.xpos = 10.0 * np.cos(angle)
        self.ypos = 10.0 * np.sin(angle)