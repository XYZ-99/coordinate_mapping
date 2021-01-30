import math
import random
import torch
import os
import numpy as np
import cv2
from skimage import io
from torchvision import transforms

from generate_coords import CoordGenerator

class octree:
    # sceneA_coords and sceneB_coords should be [3, HxW] tensors
    def __init__(self, sceneA_coords, sceneB_coords, Bcoords_colors=None, Bcoords_original_colors=None, scene_name="", offset=[0, 0, 0]):
        self.tree_depth = math.ceil(math.log(sceneA_coords.shape[1], 8))
        # 0 for min, 1 for max
        self.x_range = [0] * 2
        self.y_range = [0] * 2
        self.z_range = [0] * 2
        self.tree = [None] * (1 << (3 * self.tree_depth))

        # tree_id -> [color_list]
        self.color_tree = {}
        self.original_color_tree = {}
        self.scene_name = scene_name
        self.offset = np.array(offset)

        self.x_range[0], self.y_range[0], self.z_range[0] = torch.min(sceneA_coords, dim=1)[0].numpy().tolist()
        self.x_range[1], self.y_range[1], self.z_range[1] = torch.max(sceneA_coords, dim=1)[0].numpy().tolist()

        assert sceneA_coords.shape == sceneB_coords.shape

        for i in range(sceneA_coords.shape[1]):
            # Abort invalid mapping
            if sceneB_coords[:, i].sum() == 0:
                continue

            if Bcoords_colors is None:
                self.append_node(sceneA_coords[:, i].numpy().tolist(),
                                 sceneB_coords[:, i].numpy().tolist())
            else:
                assert Bcoords_colors.shape == Bcoords_original_colors.shape
                self.append_color_node(sceneA_coords[:, i].numpy().tolist(),
                                       sceneB_coords[:, i].numpy().tolist(),
                                       Bcoords_colors[:, i].numpy().tolist(),
                                       Bcoords_original_colors[:, i].numpy().tolist())

    # both sceneA_coord and sceneB_coord should be 3D vectors
    def append_node(self, sceneA_coord, sceneB_coord):
        tree_id = self.coord2ID(sceneA_coord)
        if self.tree[tree_id] is None:
            self.tree[tree_id] = [sceneB_coord]
        else:
            self.tree[tree_id].append(sceneB_coord)

    def append_color_node(self, sceneA_coord, sceneB_coord, Bcoord_color, Bcoord_original_color):
        tree_id = self.coord2ID(sceneA_coord)
        if self.tree[tree_id] is None:
            self.tree[tree_id] = [sceneB_coord]
            self.color_tree[tree_id] = [Bcoord_color]
            self.original_color_tree[tree_id] = [Bcoord_original_color]
        else:
            self.tree[tree_id].append(sceneB_coord)
            self.color_tree[tree_id].append(Bcoord_color)
            self.original_color_tree[tree_id].append(Bcoord_original_color)

    # coord should be a 3D vector
    # returns the ID in the tree
    def coord2ID(self, coord):
        x_min, y_min, z_min = self.x_range[0], self.y_range[0], self.z_range[0]
        x_max, y_max, z_max = self.x_range[1], self.y_range[1], self.z_range[1]

        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        z_mid = (z_min + z_min) / 2

        if coord[0] <= x_mid:
            x_id = 0
            x_max = x_mid
        else:
            x_id = 1
            x_min = x_mid

        if coord[1] <= y_mid:
            y_id = 0
            y_max = y_mid
        else:
            y_id = 1
            y_min = y_mid

        if coord[2] <= z_mid:
            z_id = 0
            z_max = z_mid
        else:
            z_id = 1
            z_min = z_mid

        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        z_mid = (z_min + z_min) / 2

        for i in range(self.tree_depth - 1):
            x_id <<= 1
            y_id <<= 1
            z_id <<= 1

            if coord[0] <= x_mid:
                x_max = x_mid
            else:
                x_id |= 1
                x_min = x_mid

            if coord[1] <= y_mid:
                y_max = y_mid
            else:
                y_id |= 1
                y_min = y_mid

            if coord[2] <= z_mid:
                z_max = z_mid
            else:
                z_id |= 1
                z_min = z_mid

            x_mid = (x_min + x_max) / 2
            y_mid = (y_min + y_max) / 2
            z_mid = (z_min + z_min) / 2

        id = (x_id << (2 * self.tree_depth)) | (y_id << self.tree_depth) | z_id
        return id

    # find the non-empty nearest neighbor box
    # returns the NN's id
    def find_NN_by_id(self, tree_id):
        x_id = tree_id >> (2 * self.tree_depth)
        y_id = (tree_id >> self.tree_depth) & ((1 << self.tree_depth) - 1)
        z_id = tree_id & ((1 << self.tree_depth) - 1)

        for search_distance in range(1, 1 << self.tree_depth):
            for delta_x in range(3 * search_distance + 1):
                for delta_y in range(3 * search_distance - delta_x + 1):
                    delta_z = 3 * search_distance - delta_x - delta_y
                    if delta_z < 0:
                        continue
                    # +++ ++- +-+ +-- -++ -+- --+ ---
                    # 1. check for boundary
                    # 2. construct the id
                    # 3. not empty
                    # 4. return
                    boundary = 1 << self.tree_depth
                    if x_id + delta_x < boundary and \
                       y_id + delta_y < boundary and \
                       z_id + delta_z < boundary:
                        nn_id = ((x_id + delta_x) << (2 * self.tree_depth)) | ((y_id + delta_y) << self.tree_depth) | (z_id + delta_z)
                        if self.tree[nn_id] is not None:
                            return nn_id

                    if x_id + delta_x < boundary and \
                       y_id + delta_y < boundary and \
                       z_id - delta_z >= 0:
                        nn_id = ((x_id + delta_x) << (2 * self.tree_depth)) | ((y_id + delta_y) << self.tree_depth) | (z_id - delta_z)
                        if self.tree[nn_id] is not None:
                            return nn_id

                    if x_id + delta_x < boundary and \
                       y_id - delta_y >= 0 and \
                       z_id + delta_z < boundary:
                        nn_id = ((x_id + delta_x) << (2 * self.tree_depth)) | ((y_id - delta_y) << self.tree_depth) | (z_id + delta_z)
                        if self.tree[nn_id] is not None:
                            return nn_id

                    if x_id + delta_x < boundary and \
                       y_id - delta_y >= 0 and \
                       z_id - delta_z >= 0:
                        nn_id = ((x_id + delta_x) << (2 * self.tree_depth)) | ((y_id - delta_y) << self.tree_depth) | (z_id - delta_z)
                        if self.tree[nn_id] is not None:
                            return nn_id

                    if x_id - delta_x >= 0 and \
                       y_id + delta_y < boundary and \
                       z_id + delta_z < boundary:
                        nn_id = ((x_id - delta_x) << (2 * self.tree_depth)) | ((y_id + delta_y) << self.tree_depth) | (z_id + delta_z)
                        if self.tree[nn_id] is not None:
                            return nn_id

                    if x_id - delta_x >= 0 and \
                       y_id + delta_y < boundary and \
                       z_id - delta_z >= 0:
                        nn_id = ((x_id - delta_x) << (2 * self.tree_depth)) | ((y_id + delta_y) << self.tree_depth) | (z_id - delta_z)
                        if self.tree[nn_id] is not None:
                            return nn_id

                    if x_id - delta_x >= 0 and \
                       y_id - delta_y >= 0 and \
                       z_id + delta_z < boundary:
                        nn_id = ((x_id - delta_x) << (2 * self.tree_depth)) | ((y_id - delta_y) << self.tree_depth) | (z_id + delta_z)
                        if self.tree[nn_id] is not None:
                            return nn_id

                    if x_id - delta_x >= 0 and \
                       y_id - delta_y >= 0 and \
                       z_id - delta_z >= 0:
                        nn_id = ((x_id - delta_x) << (2 * self.tree_depth)) | ((y_id - delta_y) << self.tree_depth) | (z_id - delta_z)
                        if self.tree[nn_id] is not None:
                            return nn_id

        print("----- find_NN_by_id does not find the NN -----")
        # print("-" * 10, "find_NN_by_id does not find the NN", "-" * 10)
        return None

    # coords should be a [3, H=60, W=80] tensor
    # returns a [3, H=60, W=80] tensor
    # img_path example: ../../datasets/7scenes_stairs/test/rgb/seq-04-frame-000289.color.png
    def group_coordinate_mapping(self, coords, img_path=''):
        new_coords = torch.Tensor(coords.shape)

        if img_path != '':
            img_dir, img_name = os.path.split(img_path)
            seq = img_name[4:6]
            frame = img_name[13:19]
            source_dir = os.path.join("../../datasets/7scenes_source", self.scene_name, "seq-"+seq)

            depth_path = os.path.join(source_dir, "frame-"+frame+".depth.png")
            pose_path = os.path.join(source_dir, "frame-"+frame+".pose.txt")

            intrinsics = np.array([[525.0,   0.0, 320.0],
                                   [  0.0, 525.0, 240.0],
                                   [  0.0,   0.0,   1.0]])

            depth = cv2.imread(depth_path, -1)
            depth = depth / 1000
            pose = np.loadtxt(pose_path)

            image_width, image_height = depth.shape[1], depth.shape[0]

            cg = CoordGenerator(intrinsics, image_width, image_height)

            gt_coords, _ = cg.depth_pose_2coord(depth, pose)

        for i in range(coords.shape[1]):
            for j in range(coords.shape[2]):
                # if the coord is not in the bounding box, find the nearest neighbor
                cur_coord = coords[:, i, j].numpy()
                if img_path != '':
                    gt_coord = gt_coords[8 * i, 8 * j, :] - self.offset # size: [3]
                # new_coords[:, i, j] = self.coordinate_mapping_random(cur_coord)
                new_coords[:, i, j] = self.coordinate_mapping_upper_bound(cur_coord, gt_coord)
#                 new_coords[:, i, j] = torch.Tensor(gt_coord)

        return new_coords

    # img_path example: ../../datasets/7scenes_stairs/test/rgb/seq-04-frame-000289.color.png
    def group_color_matching_coordinate_mapping(self, coords, img_path):
        new_coords = torch.Tensor(coords.shape)

        img_dir, img_name = os.path.split(img_path)
        seq = img_name[4:6]
        frame = img_name[13:19]
        source_dir = os.path.join("../../datasets/7scenes_source", self.scene_name, "seq-"+seq)

        photo_path = os.path.join(source_dir, "frame-"+frame+".color.png")

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

        for i in range(coords.shape[1]):
            for j in range(coords.shape[2]):
                cur_coord = coords[:, i, j].numpy()
                cur_color = image[:, 8 * i, 8 * j].numpy()
                new_coords[:, i, j] = self.color_matching_coordinate_mapping(cur_coord, cur_color)

        return new_coords

    # cur_color should be a [3] np.array
    def color_matching_coordinate_mapping(self, cur_coord, cur_color):
        cur_color = cur_color.reshape([1, 3])
        cur_coord = self.shift_coord_into_bound(cur_coord)
        tree_id = self.coord2ID(cur_coord)

        if self.tree[tree_id] is not None:
            if len(self.tree[tree_id]) == 1:
                new_coord = self.tree[tree_id][0]
            else:
                color_search_space = np.array(self.color_tree[tree_id]) # [?, 3]
                cur_color = cur_color.repeat(color_search_space.shape[0], axis=0)
                distances = np.sqrt(np.sum(np.square(cur_color - color_search_space), 1))
                min_index = np.argmin(distances)
                new_coord = self.tree[tree_id][min_index]
        else:
            nn_id = self.find_NN_by_id(tree_id)
            if len(self.tree[nn_id]) == 1:
                new_coord = self.tree[nn_id][0]
            else:
                color_search_space = np.array(self.color_tree[nn_id]) # [?, 3]
                cur_color = cur_color.repeat(color_search_space.shape[0], axis=0)
                distances = np.sqrt(np.sum(np.square(cur_color - color_search_space), 1))
                min_index = np.argmin(distances)
                new_coord = self.tree[nn_id][min_index]

        return torch.Tensor(new_coord)

    # coord should be a [3] array
    def coordinate_mapping_random(self, cur_coord):
        cur_coord = self.shift_coord_into_bound(cur_coord)

        tree_id = self.coord2ID(cur_coord)

        if self.tree[tree_id] is not None:
            random_selector = random.randrange(len(self.tree[tree_id]))
            new_coord = self.tree[tree_id][random_selector]

            # new_coord = torch.mean(torch.Tensor(self.tree[tree_id]), 0)
        else:
            nn_id = self.find_NN_by_id(tree_id)
            random_selector = random.randrange(len(self.tree[nn_id]))
            new_coord = self.tree[nn_id][random_selector]
            # new_coord = torch.mean(torch.Tensor(self.tree[nn_id]), 0)

        return new_coord

    def coordinate_mapping_upper_bound(self, cur_coord, gt_coord):
        gt_coord = gt_coord.reshape([1, 3])
        cur_coord = self.shift_coord_into_bound(cur_coord)

        tree_id = self.coord2ID(cur_coord)

        if self.tree[tree_id] is not None:
            if len(self.tree[tree_id]) == 1:
                new_coord = self.tree[tree_id][0]
            else:
                # search NN in self.tree[tree_id]
                search_space = np.array(self.tree[tree_id]) # [?, 3]
                gt_coord = gt_coord.repeat(search_space.shape[0], axis=0)
                distances = np.sqrt(np.sum(np.square(gt_coord - search_space), 1))
                min_index = np.argmin(distances)
                new_coord = self.tree[tree_id][min_index]
        else:
            nn_id = self.find_NN_by_id(tree_id)
            if len(self.tree[nn_id]) == 1:
                new_coord = self.tree[nn_id][0]
            else:
                # search NN in self.tree[nn_id]
                search_space = np.array(self.tree[nn_id]) # [?, 3]
                gt_coord = gt_coord.repeat(search_space.shape[0], axis=0)
                distances = np.sqrt(np.sum(np.square(gt_coord - search_space), 1))
                min_index = np.argmin(distances)
                new_coord = self.tree[nn_id][min_index]
        return torch.Tensor(new_coord)

    def shift_coord_into_bound(self, cur_coord):
        if cur_coord[0] < self.x_range[0]:
            cur_coord[0] = self.x_range[0]
        elif cur_coord[0] > self.x_range[1]:
            cur_coord[0] = self.x_range[1]

        if cur_coord[1] < self.y_range[0]:
            cur_coord[1] = self.y_range[0]
        elif cur_coord[1] > self.y_range[1]:
            cur_coord[1] = self.y_range[1]

        if cur_coord[2] < self.z_range[0]:
            cur_coord[2] = self.z_range[0]
        elif cur_coord[2] > self.z_range[1]:
            cur_coord[2] = self.z_range[1]

        return cur_coord
