from octree import octree
import torch
import pickle
import numpy as np
import math

with open("my_tree.txt", "rb") as f:
    my_tree: octree = pickle.load(f)

print("my_tree.tree_depth: ", my_tree.tree_depth)

print("my_tree tree size: ", len(my_tree.tree))
print("my_tree tree x_range: ", my_tree.x_range)
print("my_tree tree y_range: ", my_tree.y_range)
print("my_tree tree z_range: ", my_tree.z_range)


with open("sceneA_coords.txt", "rb") as f:
    sceneA_coords = pickle.load(f)

with open("sceneB_coords.txt", "rb") as f:
    sceneB_coords = pickle.load(f)

Bx_range = [0] * 2
By_range = [0] * 2
Bz_range = [0] * 2

Bx_range[0], By_range[0], Bz_range[0] = torch.min(sceneB_coords, dim=1)[0].detach().numpy().tolist()
Bx_range[1], By_range[1], Bz_range[1] = torch.max(sceneB_coords, dim=1)[0].detach().numpy().tolist()
print("\nBx_range: ", Bx_range)
print("By_range: ", By_range)
print("Bz_range: ", Bz_range, "\n")

assert sceneA_coords.shape == sceneB_coords.shape
coord1 = np.empty(shape=[0, 3]) # should be [n, 3]
coord2 = np.empty(shape=[0, 3])
for i in range(sceneA_coords.shape[1]):
    if sceneB_coords[:, i].sum(0) == 0:
        continue

    new_sceneB_coord = my_tree.coordinate_mapping_upper_bound(sceneA_coords[:, i], sceneB_coords[:, i].detach().numpy())
    coord1 = np.concatenate((coord1, torch.unsqueeze(sceneB_coords[:, i], 0).numpy()))
    coord2 = np.concatenate((coord2, torch.unsqueeze(new_sceneB_coord, 0).numpy()))

assert coord1.shape[1] == 3
distances = np.sqrt(np.sum(np.square(coord1 - coord2), 1))

print("distances median: ", np.median(distances))
print("distances shape: ", distances.shape)
