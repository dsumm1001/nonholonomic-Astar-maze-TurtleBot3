import numpy as np
import cv2

mazeSize = (250, 600)
clearance = 10

# Create blank maze
maze = np.zeros((mazeSize[0], mazeSize[1], 3), dtype=np.uint8)
maze[:] = (0, 255, 0)

# Draw rectangular obstacles
cv2.rectangle(
    maze,
    pt1=(clearance, clearance),
    pt2=(mazeSize[1] - clearance, mazeSize[0] - clearance),
    color=(255, 255, 255),
    thickness=-1,
)

cv2.rectangle(
    maze,
    pt1=(150 - clearance, 0),
    pt2=(165 + clearance, 125 + clearance),
    color=(0, 255, 0),
    thickness=-1,
)

cv2.rectangle(
    maze,
    pt1=(250 - clearance, 125 -clearance),
    pt2=(265 + clearance, 250 + clearance),
    color=(0, 255, 0),
    thickness=-1,
)

cv2.rectangle(maze, pt1=(150, 0), pt2=(165, 125), color=(0, 0, 255), thickness=-1)
cv2.rectangle(
    maze, pt1=(250, 125), pt2=(265, 250), color=(0, 0, 255), thickness=-1
)

# Draw triangular boundary
cv2.circle(maze, (400, 110 ), (50 + clearance), color=(0, 255, 0), thickness=-1)
cv2.circle(maze, (400, 110), 50, color=(0, 0, 255), thickness=-1)
# Show the maze image
cv2.imshow("Maze", maze)
cv2.waitKey(0)
cv2.destroyAllWindows()
