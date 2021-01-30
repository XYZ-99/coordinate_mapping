import torch
import os
from skimage import io
from torchvision import transforms
import pickle
import argparse
import numpy as np
import cv2

from octree import octree
from expert_ensemble import ExpertEnsemble
from generate_coords import CoordGenerator


parser = argparse.ArgumentParser(
    description='Train an octree to map the coordinates from an trained scene to an untrained scene.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--append', '-apd', default='', help='strings to append to models to distinguish')

parser.add_argument('--tree', default='', help='custom tree name.')

opt = parser.parse_args()

my_gating_capacity = 1
if opt.append == "chess":
    my_gating_capacity = 1
elif opt.append == "office":
    my_gating_capacity = 2

print("Loading ensemble...")
ensemble = ExpertEnsemble(1, gating_capacity=my_gating_capacity)
ensemble.load_ensemble("esac_%s.net" % opt.append)
ensemble.eval()
print("Ensemble loaded.")

with open("env_list_original.txt", "r+") as f:
    environment = f.readlines()

assert len(environment) == 7, "ERROR: The number of lines in env_list_original.txt is {:d}".format(len(environment))
scene2linenum = {
    "chess": 0,
    "fire": 1,
    "heads": 2,
    "office": 3,
    "pumpkin": 4,
    "redkitchen": 5,
    "stairs": 6
}
environment = environment[scene2linenum[opt.tree]].split()
scene = environment[0]
print("Environment: ", scene)

means = [0, 0, 0]

if len(environment) > 1:
    means[0] = float(environment[1])
    means[1] = float(environment[2])
    means[2] = float(environment[3])
print("Coordinate offsets: ", means)

scene_name = scene[scene.find("7scenes_") + 8:]
print("Scene Name: ", scene_name)

# seqs = [1, 3, 4, 5, 8, 10]
# frame_num = 1000

sceneA_coords = torch.Tensor()
sceneB_coords = torch.Tensor()
Bcoords_colors = torch.Tensor()
Bcoords_original_colors = torch.Tensor()

root_dir = os.path.join("../../datasets/data", scene_name)

intrinsics = np.array([[525.0,   0.0, 320.0],
                       [  0.0, 525.0, 240.0],
                       [  0.0,   0.0,   1.0]])

frame_num = [14, 10, 11, 21, 12, 21, 16]

for frame in range(frame_num[scene2linenum[opt.tree]]):
    file_prefix = "frame-{:06d}".format(frame)
    print("Processing file: ", file_prefix)

    photo_path = os.path.join(root_dir, file_prefix + ".color.png")
    pose_path = os.path.join(root_dir, file_prefix + ".pose.txt")
    depth_path = os.path.join(root_dir, file_prefix + ".depth.png")

    print("Reading from", photo_path)

    # --------- for sceneA_coords ------------
    image = io.imread(photo_path)
    image2 = torch.from_numpy(image.transpose([2, 0, 1])).float()
    for i in range(0, image2.shape[1], 8):
        for j in range(0, image2.shape[2], 8):
            Bcoord_original_color = image2[:, i, j].view(3, -1)
            Bcoords_original_colors = torch.cat((Bcoords_original_colors, Bcoord_original_color), 1)

    image_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(480),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4, 0.4, 0.4],
            std=[0.25, 0.25, 0.25]
            )
        ])
    # a [3, 480, 640] tensor
    image = image_transform(image)

    # construct Bcoords_colors
    for i in range(0, image.shape[1], 8):
        for j in range(0, image.shape[2], 8):
            Bcoord_color = image[:, i, j].view(3, -1)
            Bcoords_colors = torch.cat((Bcoords_colors, Bcoord_color), 1)

    image = image.unsqueeze(0)
    image = image.cuda()

    prediction = ensemble.scene_coordinates(0, image)

    # [1, 3, 60, 80]
    print("prediction size: ", prediction.shape)
    prediction = prediction.cpu()
    sceneA_coords = torch.cat((sceneA_coords, prediction.view(3, -1)), 1)

    # ---------- for sceneB_coords -----------
    depth = cv2.imread(depth_path, -1)
    depth = depth / 1000
    pose = np.loadtxt(pose_path)

    image_width, image_height = depth.shape[1], depth.shape[0]

    cg = CoordGenerator(intrinsics, image_width, image_height)

    gt_coords, _ = cg.depth_pose_2coord(depth, pose)

    offset = torch.Tensor(means)
    offset = offset.reshape([3, 1])

    for i in range(0, gt_coords.shape[0], 8):
        for j in range(0, gt_coords.shape[1], 8):
            sample_coord = gt_coords[i, j, :]
            sample_coord = torch.from_numpy(sample_coord.reshape([3, 1]).astype("float32"))
            if sample_coord.abs().sum() != 0:
                sample_coord -= offset
            sceneB_coords = torch.cat((sceneB_coords, sample_coord), 1)


print("sceneA_coords size: ", sceneA_coords.shape)
print("sceneB_coords size: ", sceneB_coords.shape)
print("Bcoords_colors size: ", Bcoords_colors.shape)
print("Bcoords_original_colors size: ", Bcoords_original_colors.shape)
print("offset: ", means)
my_tree = octree(sceneA_coords.detach(), sceneB_coords.detach(), Bcoords_colors.detach(),
                 Bcoords_original_colors.detach(), scene_name=scene_name, offset=means)
print("My tree length: ", len(my_tree.tree))
print("My color tree length: ", len(my_tree.color_tree))
print("My original color tree length ", len(my_tree.original_color_tree))

with open("sceneA_coords.txt", "wb") as f:
    pickle.dump(sceneA_coords, f)

with open("sceneB_coords.txt", "wb") as f:
    pickle.dump(sceneB_coords, f)

# save the octree
with open("my_tree_color_"+opt.append+"_"+opt.tree+".txt", "wb") as f:
    pickle.dump(my_tree, f)
print("Octree successfully saved as my_tree_color_"+opt.append+"_"+opt.tree+".txt.")
