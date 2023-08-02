import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

poses_file = "/home/chrisliu/Projects/ActiveRecon/ActiveRecon/exp/realworld_6_box/poses.txt"
poses = np.loadtxt(poses_file)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for pose in poses:
    pose = pose.reshape(4, 4)
    print(pose)
    ax.quiver(
        pose[0, 3],
        pose[1, 3],
        pose[2, 3],
        pose[0, 0],
        pose[1, 0],
        pose[2, 0],
        length=0.1,
        color="r",
    )
    ax.quiver(
        pose[0, 3],
        pose[1, 3],
        pose[2, 3],
        pose[0, 1],
        pose[1, 1],
        pose[2, 1],
        length=0.1,
        color="g",
    )
    ax.quiver(
        pose[0, 3],
        pose[1, 3],
        pose[2, 3],
        pose[0, 2],
        pose[1, 2],
        pose[2, 2],
        length=0.1,
        color="b",
    )

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
size = 0.5
ax.set_xlim(-size, size)
ax.set_ylim(-size, size)
ax.set_zlim(-size, size)
plt.show()
