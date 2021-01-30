from skimage import io
import torch
import numpy as np

img = io.imread("../../datasets/7scenes_source/fire/seq-01/frame-000000.depth.png")
img = np.array(img).astype("float64")


with open("../../datasets/7scenes_source/fire/seq-01/frame-000000.pose.txt", "r+") as f:
    lines = f.readlines()

lines = [line.split() for line in lines]
for i in range(len(lines)):
    for j in range(len(lines[0])):
        lines[i][j] = float(lines[i][j])

extrinsics = np.array(lines)
R = extrinsics[0:3, 0:3]
t = extrinsics[0:3, 3]

intrinsics = np.array([[525.0,   0.0, 320.0],
                       [  0.0, 525.0, 240.0],
                       [  0.0,   0.0,   1.0]])

x, y = np.meshgrid(np.arange(0, img.shape[1]), np.arange(0, img.shape[0]))
x, y = x.reshape([-1]), y.reshape([-1])

# np.vstack((x, y, np.ones_like(x))) * img.reshape([-1])
print(t.reshape([3, 1]).repeat(x.shape[0], axis=1).shape)

before_rotation = np.matmul(np.linalg.inv(intrinsics), np.vstack((x, y, np.ones_like(x))) * img.reshape([-1])) \
                  - t.reshape([3, 1]).repeat(x.shape[0], axis=1)
# world_coord = np.matmul( , )

