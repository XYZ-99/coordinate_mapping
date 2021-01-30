import os
import pickle
import torch
import argparse
from skimage import io
from torchvision import transforms
import cv2
import numpy as np

from octree import octree
from expert_ensemble import ExpertEnsemble
from generate_coords import CoordGenerator


parser = argparse.ArgumentParser(
    description='Train an octree to map the coordinates from an trained scene to an untrained scene.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--session', '-sid', default='',
    help='custom session name appended to output files, useful to separate different runs of a script')

parser.add_argument('--tree', default='',
                    help='custom tree name.')

opt = parser.parse_args()

my_gating_capacity = 1
if opt.session == "chess":
    my_gating_capacity = 1
elif opt.session == "office":
    my_gating_capacity = 2

print("Loading ensemble...")
ensemble = ExpertEnsemble(1, gating_capacity=my_gating_capacity)
ensemble.load_ensemble("esac_%s.net" % opt.session)
ensemble.eval()
print("Ensemble loaded.")

print("Opening octree...")
with open("my_tree_color_{}_{}.txt".format(opt.session, opt.tree), "rb") as f:
    my_tree: octree = pickle.load(f)
print("Octree loaded.")

testing_seq = {
    "chess": [3, 5],
    "fire": [4], # should be [3, 4]
    "heads": [1],
    "office": [2, 6, 7, 9],
    "pumpkin": [1, 7],
    "redkitchen": [3, 4, 6, 12, 14],
    "stairs": [1, 4]
}

for seq in testing_seq[opt.tree]:
    root_dir = os.path.join("../../datasets/7scenes_source", opt.tree, "seq-{:02d}".format(seq))
    print("Entering", opt.tree, seq)

    # ------------------ Testing code below -------------------------
    frame_num = 1000
    if opt.tree == "stairs":
        frame_num = 500

    for frame in range(0, frame_num, 10): # TODO: check the number
        filename = "frame-{:06d}".format(frame)
        photo_path = os.path.join(root_dir, filename+".color.png")
        print("Now processing: ", photo_path)

        image = io.imread(photo_path)

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

        image = image.unsqueeze(0)
        image = image.cuda()

        # a [1, 3, 60, 80] tensor
        prediction = ensemble.scene_coordinates(0, image)

        # a [3, 60, 80] tensor
        prediction = prediction.squeeze(0)

        output_path = "./7scenes_" + opt.session + "_" + opt.tree + "/seq{:02d}".format(seq)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(os.path.join(output_path, filename+".pixels.txt"), "w+") as pixelFile:
            pixelFile.write("4800\n")

            for i in range(0, 480, 8):
                for j in range(0, 640, 8):
                    pixelFile.write("{} {}\n".format(j, i))
        print("Written to", os.path.join(output_path, filename+".pixels.txt"))

        # ----------------------- final file format below ----------------------------
        coordFile = open(os.path.join(output_path, filename+".coords.txt"), "w+")
        colorFile = open(os.path.join(output_path, filename+".colors.txt"), "w+")

        coordFile.write("4800\n")
        colorFile.write("4800\n")

        colorwrite = 0
        coordwrite = 0

        for i in range(60):
            for j in range(80):
                cur_coord = prediction[:, i, j].cpu().detach().numpy()

                # cur_coord should be a [3] np.array
                assert(cur_coord.shape[0] == 3)

                cur_coord = my_tree.shift_coord_into_bound(cur_coord)

                tree_id = my_tree.coord2ID(cur_coord)

                id = tree_id if my_tree.tree[tree_id] is not None else my_tree.find_NN_by_id(tree_id)

                coordFile.write( ("%d\n" % len(my_tree.tree[id])) )
                colorFile.write( ("%d\n" % len(my_tree.original_color_tree[id])) )
                assert len(my_tree.tree[id]) == len(my_tree.original_color_tree[id])
                for k in range(len(my_tree.tree[id])):
                    colorFile.write("{} {} {}\n".format(my_tree.original_color_tree[id][k][0],
                                                        my_tree.original_color_tree[id][k][1],
                                                        my_tree.original_color_tree[id][k][2]))
                    colorwrite += 1
                    coordFile.write("{} {} {}\n".format(my_tree.tree[id][k][0],
                                                        my_tree.tree[id][k][1],
                                                        my_tree.tree[id][k][2]))
                    coordwrite += 1

        print("Written to", os.path.join(output_path, filename+".coords.txt"))
        print("Written to", os.path.join(output_path, filename+".colors.txt"))
        print("colorwrite: ", colorwrite)
        print("coordwrite: ", coordwrite)
        coordFile.close()
        colorFile.close()
        # ---------------------- final file format above -----------------------



        # ---------------------- Upper bound test below -----------------------
        # coordFile = open("./7scenes_fire/seq03/"+filename+".coords.txt", "w+")
        # colorFile = open("./7scenes_fire/seq03/"+filename+".colors.txt", "w+")
        #
        # coordFile.write("4800\n")
        # colorFile.write("4800\n")
        #
        # new_coords = torch.Tensor(prediction.shape)
        #
        # depth_path = os.path.join(root_dir, filename+".depth.png")
        # pose_path = os.path.join(root_dir, filename+".pose.txt")
        #
        # intrinsics = np.array([[525.0,   0.0, 320.0],
        #                        [  0.0, 525.0, 240.0],
        #                        [  0.0,   0.0,   1.0]])
        #
        # depth = cv2.imread(depth_path, -1)
        # depth = depth / 1000
        # pose = np.loadtxt(pose_path)
        #
        # image_width, image_height = depth.shape[1], depth.shape[0]
        # # print("Image Height: ", image_height)
        # # print("Image width: ", image_width)
        #
        # cg = CoordGenerator(intrinsics, image_width, image_height)
        #
        # gt_coords, _ = cg.depth_pose_2coord(depth, pose)
        # # print("gt_coords size: ", gt_coords.shape)
        #
        # for i in range(prediction.shape[1]):
        #     for j in range(prediction.shape[2]):
        #         cur_coord = prediction[:, i, j].cpu().detach().numpy()
        #
        #         gt_coord = gt_coords[8 * i, 8 * j, :]
        #         gt_coord = gt_coord.reshape([1, 3])
        #         cur_coord = my_tree.shift_coord_into_bound(cur_coord)
        #
        #         tree_id = my_tree.coord2ID(cur_coord)
        #
        #         id = tree_id if my_tree.tree[tree_id] is not None else my_tree.find_NN_by_id(tree_id)
        #
        #         if len(my_tree.tree[id]) == 1:
        #             coordFile.write("1\n{} {} {}\n".format(my_tree.tree[id][0][0],
        #                                                    my_tree.tree[id][0][1],
        #                                                    my_tree.tree[id][0][2]))
        #             colorFile.write("1\n{} {} {}\n".format(my_tree.original_color_tree[id][0][0],
        #                                                    my_tree.original_color_tree[id][0][1],
        #                                                    my_tree.original_color_tree[id][0][2]))
        #
        #         else:
        #             search_space = np.array(my_tree.tree[id])
        #             gt_coord = gt_coord.repeat(search_space.shape[0], axis=0)
        #             distances = np.sqrt(np.sum(np.square(gt_coord - search_space), 1))
        #             min_index = np.argmin(distances)
        #             coordFile.write("1\n{} {} {}\n".format(my_tree.tree[id][min_index][0],
        #                                                    my_tree.tree[id][min_index][1],
        #                                                    my_tree.tree[id][min_index][2]))
        #
        #             colorFile.write("1\n{} {} {}\n".format(my_tree.original_color_tree[id][min_index][0],
        #                                                    my_tree.original_color_tree[id][min_index][1],
        #                                                    my_tree.original_color_tree[id][min_index][2]))

        # ---------------------- Upper bound test above -----------------------

        # ---------------------- Real Ground Truth below ----------------------
        # coordFile = open("./7scenes_fire/seq03/"+filename+".coords.txt", "w+")
        # colorFile = open("./7scenes_fire/seq03/"+filename+".colors.txt", "w+")
        #
        # coordFile.write("4800\n")
        # colorFile.write("4800\n")
        #
        # depth_path = os.path.join(root_dir, filename+".depth.png")
        # pose_path = os.path.join(root_dir, filename+".pose.txt")
        #
        #
        #
        # intrinsics = np.array([[525.0,   0.0, 320.0],
        #                        [  0.0, 525.0, 240.0],
        #                        [  0.0,   0.0,   1.0]])
        #
        # depth = cv2.imread(depth_path, -1)
        # depth = depth / 1000
        # pose = np.loadtxt(pose_path)
        #
        # image_width, image_height = depth.shape[1], depth.shape[0]
        # # print("Image Height: ", image_height)
        # # print("Image width: ", image_width)
        #
        # cg = CoordGenerator(intrinsics, image_width, image_height)
        #
        # gt_coords, _ = cg.depth_pose_2coord(depth, pose)
        # # (480, 640, 3)
        # # print("gt_coords size: ", gt_coords.shape)
        #
        # # (480, 640, 3)
        # colors = cv2.imread(photo_path)
        # # print("colors size: ", colors.shape)
        #
        # for i in range(0, 480, 8):
        #     for j in range(0, 640, 8):
        #         coordFile.write("1\n{} {} {}\n".format(gt_coords[i, j, :][0],
        #                                                gt_coords[i, j, :][1],
        #                                                gt_coords[i, j, :][2]))
        #
        #         colorFile.write("1\n{} {} {}\n".format(colors[i, j, :][0],
        #                                                colors[i, j, :][1],
        #                                                colors[i, j, :][2]))
        #
        # coordFile.close()
        # colorFile.close()

    # ---------------------- Real Ground Truth above ----------------------

    # ------------------ Testing code above -------------------------