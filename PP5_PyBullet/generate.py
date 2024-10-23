import pyrosim.pyrosim as ps

# Global parameters
l = 1 # length
w = 1 # width 
h = 1 # height 

x = 0
y = 0
z = 0.5

def Create_World():
    ps.Start_SDF("box.sdf")
    for i in range(10):
        ps.Send_Cube(name="Box",pos=[x,y,z],size=[l,w,h])
        z += h
        l = 0.9 * l
        w = 0.9 * w
        h = 0.9 * h
    ps.End()

def Create_Robot():
    ps.Start_URDF("body.urdf")

    # Define sizes and positions for the cubes (1 unit size)
    cube_size = [1.0, 1.0, 1.0]  # Size for each cube

    # Create the first cube (head)
    ps.Send_Cube(name="Cube1", pos=[-2.0, 0.0, 0.0], size=cube_size)

    # Create the second cube (middle)
    ps.Send_Joint(name="Cube1_Cube2", parent="Cube1", child="Cube2", type="revolute", position=[1.0, 0.0, 0.0]) 
    ps.Send_Cube(name="Cube2", pos=[-1.0, 0.0, 0.0], size=cube_size) 

    # Create the third cube (tail)
    ps.Send_Joint(name="Cube2_Cube3", parent="Cube2", child="Cube3", type="revolute", position=[1.0, 0.0, 0.0]) 
    ps.Send_Cube(name="Cube3", pos=[0.0, 0.0, 0.0], size=cube_size) 

    ps.End()

# Create_World()
Create_Robot()
