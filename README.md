# Project #3 Phase 2: Weighted A* Algo Maze Search using Turtlebot3 Platform with Non-Holonomic Constraints
Author: Doug Summerlin (dsumm1001@gmail.com, dsummerl@umd.edu)  
UID: 114760753  
Directory ID: dsummerl  
Author: Vignesh Rajagopal (vickyrv570@gmail.com, vigneshr@umd.edu)  
UID: 119476192  
Directory ID: vigneshr  
ENPM661 Spring 2023: Robotic Path Planning

**Github Repo Link**: https://github.com/dsumm1001/nonholonomic-Astar-maze-TurtleBot3.git  
**Part 01 Results Link**: https://docs.google.com/document/d/10BkwTZuwsg98dj5V_ue8OGCteuYqv3J8k4jljOtG0rY/edit?usp=sharing  
**Part 02 Results Link**: https://docs.google.com/document/d/16G7GDCzAz8--KsqxgZECaE3ZS4Szf5M-12MV9PlfBPE/edit?usp=sharing

This project is basic visual implementation of the A* algorithm for a turtlebot3 burger robot navigating a maze in the gazebo simulation environment under non-holonomic constraints. The project is accompanied by a 2D representation of the generated path and 2D simulation before using a ROS package to simulate the turlebot3. 



## PART 01:   

### Filesystems
cd into the folder titled "proj3p2\_douglas\_vignesh" after downloading it on an Ubuntu or Windows operating system. 
Nothing else should be required here, the code relies on no external files. 

Ensure the following libraries are installed so they can be imported as such:  
`import numpy as np`  
`import matplotlib.pyplot as plt`  
`import cv2`  
`import math`  
`from queue import PriorityQueue`  
`import time`  
`import sys`
`from collections import OrderedDict`  

### OPERATING SYSTEM:
Ubuntu 20.04 Linux (but Windows 11 should also work)

### TO RUN SCRIPT:   
`\$ cd ~/home/.../Proj3\_douglas\_vignesh/Part 01`  
`\$ python3 turtlebot3\_astar\_douglas\_vignesh.py`

### PROGRAM OUTPUT:
The program will begin by prompting the user for the desired boundary clearance in units of [mm] (the map in part 01 is 6m x 2.5m) as well as the two sets of input RPMs for the turtlebot3 drive wheels. For the maze in part 01 the boundary clearance range is effectively limited to 110-120mm, to account for the geometry of the maze and the size of the turtlebot. Then the program will prompt the user start and goal node coordinates in x,y format (in [cm], origin at bottom left of map), as well as the orientation of the robot at its starting position. Please enter integer numbers between that are within the mazespace and not within the boundary or obstacle space for x and y coordinates, separated only be a comma and no whitespace, eg. "589,239" for a boundary width of 11 is acceptable. Please enter the orientation as an integer angle value between 0-360 degrees within increments as small as 1 deg. The program will repeat the prompt for the position and orientation if acceptable values are not provided. 

The program then will use a weighted A* algorithm to find the optimal path from the established start node to the established goal node using the non-holonomic 8 action set dictated. Once the optimal path has been found, the program will prompt:

*"Yay! Goal node located... Operation took  X.X seconds."*

Typical pathing time is between 1 and 5 seconds depending on the machine. The program will display the total number of seearched nodes as well as the number of total nodes in the generated path. The program will then display the generated path on an image window imposed on the maze, with all of the nonholonomic search trajectories appearing as blue arcs. The nodes representing the generated path will appear as small yellow circles. Close this window to continue.

The program will then prepare a visual simulation of the search process depicting searched areas with blue arrows, generate the optimal pathline as a series of yellow circles, and then simulate a (purple) circular mobile robot following the optimal path. This simulation will be generated and saved as an .mp4 file, before finally appearing as a video window on the OS GUI. Once the video is closed or the video ends, the program will terminate.

### INPUTS FOR PART 01:
User inputs in bold italics

The radius of the Turtlebot3 burger model is approximately  105.0  [mm].  
Please enter the desired obstacle clearance as an integer value between 105.0 and 120 [mm]: ***110***   
Enter two wheel RPMs [rev per minute] as integer values between 1 - 200 , separated by a comma: ***50,75***   
Enter start node coordinates in x, y format, in [cm], separated by a comma: ***11,11***  
Enter start node orientation as an integer between 0-359, using increments of 1 deg: ***45***  
Enter goal node coordinates in x, y format, in [cm], separated by a comma: ***589,11***  



## PART 02:   

### Filesystems
Open the folder titled "proj3p2\_douglas\_vignesh" after downloading it on an Ubuntu or Windows operating system.   
Copy the folder titled "part02" into the `catkin\_ws` on your machine, which has ROS noetic installed.

Ensure the following libraries are installed so they can be imported as such:  
`import rospy`  
`from geometry_msgs.msg import Twist`  
`import numpy as np`  
`import matplotlib.pyplot as plt`  
`import cv2`  
`import math`  
`from queue import PriorityQueue`  
`import time`  
`import sys`
`from collections import OrderedDict`  
`import tf.transformations as tf`
`from gazebo_msgs.msg import ModelState`
`from gazebo_msgs.srv import SetModelState`
`from geometry_msgs.msg import Pose, Quaternion`

### OPERATING SYSTEM:
Ubuntu 20.04 Linux (but Windows 11 should also work)

### TO RUN SCRIPT:   
In one terminal, cd into your `catkin\_ws` and run `\$ catkin_make` and `\$ source devel/setup.bash`
Then run the command `\$ roslaunch part02 part02_map.launch`. This should launch the map environment in gazebo
In a second terminal cd into `\$ ~/catkin_ws/src/part02/src`.
In the second terminal, run `\$ python3 turtlebot3\_astar\_douglas\_vignesh\_part02.py`

### PROGRAM OUTPUT:
The program will begin by prompting the user for the desired boundary clearance in units of [mm] (the map in part 02 is 6m x 2.0m) as well as the two sets of input RPMs for the turtlebot3 drive wheels. For the maze in part 01 the boundary clearance range is limited to 110-250mm. Then the program will prompt the user start and goal node coordinates in x,y format (in [cm], origin at bottom left of map), as well as the orientation of the robot at its starting position. Please enter integer numbers between that are within the mazespace and not within the boundary or obstacle space for x and y coordinates, separated only be a comma and no whitespace, eg. "589,189" for a boundary width of 11 is acceptable. Please enter the orientation as an integer angle value between 0-360 degrees within increments as small as 1 deg. The program will repeat the prompt for the position and orientation if acceptable values are not provided. 

The program then will use a weighted A* algorithm to find the optimal path from the established start node to the established goal node using the non-holonomic 8 action set dictated. Once the optimal path has been found, the program will prompt:

*"Yay! Goal node located... Operation took  X.X seconds."*

Typical pathing time is between 1 and 5 seconds depending on the machine. The program will display the total number of seearched nodes as well as the number of total nodes in the generated path. The program will then display the generated path on an image window imposed on the maze, with all of the nonholonomic search trajectories appearing as blue arcs. The nodes representing the generated path will appear as small yellow circles. Close this window to continue.

The program will then prepare a visual simulation of the search process depicting searched areas with blue arrows, generate the optimal pathline as a series of yellow circles, and then simulate a (purple) circular mobile robot following the optimal path. This simulation will be generated and saved as an .mp4 file, before finally appearing as a video window on the OS GUI. Once the video has finished playing, the window will close and the spawned turtlebot3 in the gazebo environment will assume the starting position and orientation as previously given by the user.  

Once the starting position is assumed, the terminal will print *"Position Set!"*. The program will then prompt the user for an input to begin the ROS simulation in gazebo, saying:  

*"Press enter to articulate the robot! "*  

Press enter in the terminal to begin the simulation. The turtlebot3 will follow the generated path with relative accuracy, but because there is no positonal feedback the robot may stray signficantly from its path depending on clearance input, RPM input, and path complexity. Once the path has been completely followed, the python program will terminate, finally saying *"Finished articulating Turtlebot! Program Termination."*

### INPUTS FOR PART 02:
User inputs in bold italics  

The radius of the Turtlebot3 burger model is approximately  105.0  [mm].  
Please enter the desired obstacle clearance as an integer value between 105.0 and 250 [mm]: ***150***  
Enter two wheel RPMs [rev per minute] as integer values between 1 - 200 , separated by a comma: ***25,50***  
Enter start node coordinates in x, y format, in [cm], separated by a comma: ***30,30***  
Enter start node orientation as an integer between 0-359, using increments of 1 deg: ***30***  
Enter goal node coordinates in x, y format, in [cm], separated by a comma: ***200,120***  

