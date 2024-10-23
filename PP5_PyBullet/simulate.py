import pybullet as p
import pybullet_data
import pyrosim.pyrosim as ps
import numpy as np
import time 

physicsClient = p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0,0,-9.8)
planeId = p.loadURDF("plane.urdf")
# p.loadSDF("box.sdf")
robotId = p.loadURDF("body.urdf")

duration = 10000
fixed_steps = 200

ps.Prepare_To_Simulate(robotId)

x = np.linspace(0,10*np.pi, duration)
y = np.sin(x)*np.pi/2

for i in range(duration):
    if (i // fixed_steps) % 2 == 0:  # Move Cube1 and Cube2
        ps.Set_Motor_For_Joint(bodyIndex=robotId, 
                                jointName=b'Cube1_Cube2', 
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=-0.5,  # Move Cube1 upwards
                                maxForce=500)
        ps.Set_Motor_For_Joint(bodyIndex=robotId, 
                                jointName=b'Cube2_Cube3',  
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=0.0,  # Keep Cube3 level
                                maxForce=500)
    else:  # Move Cube2 and Cube3
        ps.Set_Motor_For_Joint(bodyIndex=robotId, 
                                jointName=b'Cube1_Cube2',  
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=0.0,  # Return Cube1 to level
                                maxForce=500)
        ps.Set_Motor_For_Joint(bodyIndex=robotId, 
                                jointName=b'Cube2_Cube3',  
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=-0.5,  # Move Cube3 upwards
                                maxForce=500)
    p.stepSimulation()
    time.sleep(1/500)

p.disconnect()