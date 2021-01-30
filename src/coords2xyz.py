import pickle

with open("sceneA_coords.txt", "rb") as f:
    scene_coords = pickle.load(f)

scene_coords = scene_coords.t()

scene_coords = scene_coords.detach().numpy().tolist()

with open("sceneModel.xyz", "w+") as f:
    for line in scene_coords:
        f.write("{} {} {}\n".format(line[0], line[1], line[2]))
