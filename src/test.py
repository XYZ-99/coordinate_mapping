import torch

import esac

import time
import argparse
import math

from expert import Expert
from expert_ensemble import ExpertEnsemble
import util

import cv2
import numpy as np

import pickle

parser = argparse.ArgumentParser(
    description='for testing the entire model.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', '-m', default='',
    help='ensemble model file, if empty we use the default file name + the session ID')

parser.add_argument('--testinit', '-tinit', action='store_true',
    help='load individual expert networks and gating, used for testing before end-to-end training, we use the default file names + session ID')

parser.add_argument('--testrefined', '-tref', action='store_true',
    help='load individual refined expert networks and gating, used for testing before end-to-end training, we use the default file names + session ID + refined post fix')

parser.add_argument('--hypotheses', '-hyps', type=int, default=256, 
    help='number of hypotheses, i.e. number of RANSAC iterations')

parser.add_argument('--threshold', '-t', type=float, default=10, 
    help='inlier threshold in pixels')

parser.add_argument('--inlieralpha', '-ia', type=float, default=100, 
    help='alpha parameter of the soft inlier count; Controls the softness of the hypotheses score distribution; lower means softer')

parser.add_argument('--inlierbeta', '-ib', type=float, default=0.5, 
    help='beta parameter of the soft inlier count; controls the softness of the sigmoid; lower means softer')

parser.add_argument('--maxreprojection', '-maxr', type=float, default=100, 
    help='maximum reprojection error; reprojection error is clamped to this value for stability')

parser.add_argument('--rotthreshold', '-rt', type=float, default=5, 
    help='acceptance threshold of rotation error in degree')

parser.add_argument('--transthreshold', '-tt', type=float, default=5, 
    help='acceptance threshold of translation error in centimeters')

parser.add_argument('--expertselection', '-es', action='store_true',
    help='select one expert instead of distributing hypotheses')

parser.add_argument('--oracleselection', '-os', action='store_true',
    help='always select the ground truth expert')

parser.add_argument('--clusters', '-c', type=int, default=-1,
                   help='num of clusters the environment should be split into, corresponds to the number of desired experts')

parser.add_argument('--append', '-apd', default='',
    help='strings to append to models to distinguish')

parser.add_argument('--tree', default='',
                   help='custom tree name.')

opt = parser.parse_args()

if opt.clusters < 0:
    from room_dataset import RoomDataset
    testset = RoomDataset("test", training=False)
    ensemble = ExpertEnsemble(testset.num_experts)

else:
    from cluster_dataset import ClusterDataset
    testset = ClusterDataset("test", num_clusters=opt.clusters, training=False)
    ensemble = ExpertEnsemble(testset.num_experts, gating_capacity=2)
    
testset_loader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=6)

if opt.testrefined:
    ensemble.load_experts(opt.append, True)
elif opt.testinit:
    ensemble.load_experts(opt.append, False)
else:
    if len(opt.model) == 0:
        if opt.expertselection:
            opt.model = 'es_%s.net' % opt.append
        else:
            opt.model = 'esac_%s.net' % opt.append

    ensemble.load_ensemble(opt.model)

ensemble.eval()

if opt.testinit:
    opt.append = 'init_' + opt.append

if opt.testrefined:
    opt.append = 'ref_' + opt.append

if opt.expertselection:
    opt.append = 'es_' + opt.append

if opt.oracleselection:
    opt.append = 'os_' + opt.append

test_log = open('results_esac_color_%s_%s.txt' % (opt.append, opt.tree), 'w', 1)
pose_log = open('poses_esac_color_%s_%s.txt' % (opt.append, opt.tree), 'w', 1)

print('Environment has', len(testset), 'test images.')

scenes_r = [[]]
scenes_t = [[]]
scenes_c = [[]]

if opt.clusters < 0:
    for e in range(testset.num_experts-1):
        scenes_r.append([])
        scenes_t.append([])
        scenes_c.append([])

avg_active = 0
max_active = 0
avg_time = 0

# load my_tree
if opt.tree != "":
    with open("my_tree_color_"+opt.append+"_"+opt.tree+".txt", "rb") as f:
        my_tree = pickle.load(f)
    print("Loading ", "my_tree_color_"+opt.append+"_"+opt.tree+".txt")

step = 10
step_controller = 0

err_txt = open("err_gt_"+opt.append+"_"+opt.tree+".txt", "w", 1)

with torch.no_grad():	

    for idx, image, focallength, gt_pose, gt_coords, gt_expert in testset_loader:
        step_controller += 1
        if step_controller % step != 0:
            continue
        
        idx = int(idx[0])
        img_file = testset.get_file_name(idx)

        print("Processing image %d: %s\n" % (idx, img_file))
        
        # camera calibration
        focallength = float(focallength[0])
        pp_x = float(image.size(3) / 2)
        pp_y = float(image.size(2) / 2)

        gt_pose = gt_pose[0]
        gt_expert = int(gt_expert[0])

        # dimension of the expert prediction
        pred_w = math.ceil(image.size(3) / Expert.OUTPUT_SUBSAMPLE)
        pred_h = math.ceil(image.size(2) / Expert.OUTPUT_SUBSAMPLE)
        
        # prediction container to hold all expert outputs
        prediction = torch.zeros((testset.num_experts, 3, pred_h, pred_w)).cuda()
        image = image.cuda()

        start_time = time.time()

        # gating prediction
        gating_log_probs = ensemble.log_gating(image)
        gating_probs = torch.exp(gating_log_probs).cpu()

        if opt.oracleselection:
            gating_probs[0].fill_(0)
            gating_probs[0,gt_expert] = 1

        # assign hypotheses to experts
        if opt.expertselection or opt.oracleselection:
            expert = torch.multinomial(gating_probs[0], 1, replacement=True)
            e_hyps = expert.expand((opt.hypotheses))
        else:
            e_hyps = torch.multinomial(gating_probs[0], opt.hypotheses, replacement=True)
        
        # do experts prediction if they have at least one hypothesis
        e_hyps_hist = torch.histc(e_hyps.float(), bins=testset.num_experts, min=0, max=testset.num_experts-1)

        avg_active += float((e_hyps_hist > 0).sum())
        max_active = max(max_active, float((e_hyps_hist > 0).sum()))

        for e, count in enumerate(e_hyps_hist):
            if count > 0:
                prediction[e] = ensemble.scene_coordinates(e, image)
                if opt.tree != "":
                    prediction[e] = my_tree.group_coordinate_mapping(prediction[e].cpu(), img_path=img_file)
                prediction[e].cuda()

        prediction = prediction.cpu()

        out_pose = torch.zeros(4, 4).float()

        # perform pose estimation
        winning_expert = esac.forward(
            prediction,
            e_hyps, 
            out_pose, 
            0, 
            0, 
            focallength, 
            pp_x,
            pp_y,
            opt.threshold,
            opt.inlieralpha,
            opt.inlierbeta,
            opt.maxreprojection,
            Expert.OUTPUT_SUBSAMPLE)

        avg_time += time.time()-start_time

        # calculate pose errors
        t_err = float(torch.norm(gt_pose[0:3, 3] - out_pose[0:3, 3]))

        gt_R = gt_pose[0:3,0:3].numpy()
        out_R = out_pose[0:3,0:3].numpy()

        r_err = np.matmul(out_R, np.transpose(gt_R))
        r_err = cv2.Rodrigues(r_err)[0]
        r_err = np.linalg.norm(r_err) * 180 / math.pi

        print("\nrerr: %.2fdeg, terr: %.1fcm" % (r_err, t_err*100))

        print("\nTrue expert: ", int(gt_expert))
        print("Expert with max. prob.: ", int(gating_probs[0].max(0)[1]))
        print("Expert chosen: ", winning_expert, "\n")

        scenes_r[gt_expert].append(r_err)
        scenes_t[gt_expert].append(t_err * 100)
        scenes_c[gt_expert].append(int(gt_expert) == winning_expert)

        out_pose = out_pose.inverse()

        t = out_pose[0:3, 3]

        # rotation
        rot, _ = cv2.Rodrigues(out_pose[0:3,0:3].numpy())
        angle = np.linalg.norm(rot)
        axis = rot / angle
        q_w = math.cos(angle * 0.5)
        q_xyz = math.sin(angle * 0.5) * axis

        pose_log.write("%s %f %f %f %f %f %f %f\n" % (
            util.strip_file_name(img_file),
            q_w, q_xyz[0], q_xyz[1], q_xyz[2],
            float(t[0]), float(t[1]), float(t[2])))	
        err_txt.write("%s %f %f\n" % (util.strip_file_name(img_file), float(r_err), float(t_err*100)))

print("Scene - Expert.Acc. - Pose.Acc. - Med. Rerr. - Med. Terr.")
print("------------------------------------------------------------")

avg_class = 0
avg_pose = 0
avg_rot = 0
avg_trans = 0

for sceneIdx in range(len(scenes_c)):

    class_acc = sum(scenes_c[sceneIdx]) / max(len(scenes_c[sceneIdx]),1)
    avg_class += class_acc

    pose_acc = [(t_err < opt.transthreshold and r_err < opt.rotthreshold) for (t_err, r_err) in zip(scenes_t[sceneIdx], scenes_r[sceneIdx])]
    pose_acc = sum(pose_acc) / max(len(pose_acc),1)
    avg_pose += pose_acc

    def median(l):
        if len(l) == 0: 
            return 0
        l.sort()
        return l[int(len(l) / 2)]

    median_r = median(scenes_r[sceneIdx])
    avg_rot += median_r

    median_t = median(scenes_t[sceneIdx])
    avg_trans += median_t

    print("%7d %7.1f%% %10.1f%% %10.2fdeg %10.2fcm" % (sceneIdx, class_acc*100, pose_acc*100, median_r, median_t))
    test_log.write("%f %f %f %f\n" % (class_acc, pose_acc, median_r, median_t))

if opt.clusters < 0:
    print("------------------------------------------------------------")
    print("Average %7.1f%% %10.1f%% %10.2fdeg %10.2fcm" % (avg_class*100/testset.num_experts, avg_pose*100/testset.num_experts, avg_rot/testset.num_experts, avg_trans/testset.num_experts))

print('\nAvg. experts active:', avg_active / len(testset_loader))
print('Max. experts active:', max_active)

print("\nAvg. Time: %.3fs" % (avg_time / len(testset_loader)))

print('\nDone without errors.')
test_log.close()
pose_log.close()
err_txt.close()
