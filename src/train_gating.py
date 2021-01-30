import torch
import torch.nn as nn
import torch.optim as optim

import time
import argparse
import math
import random

from expert import Expert
from gating import Gating
import util

parser = argparse.ArgumentParser(
    description='train the gating network.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--learningrate', '-lr', 
    type=float, 
    default=0.0001, 
    help='learning rate')

parser.add_argument('--iterations', '-iter', 
    type=int, 
    default=100000,
    help='number of training iterations, i.e. numer of model updates')

parser.add_argument('--clusters', '-c', 
    type=int, 
    default=1,
    help='number of clusters the environment should be split into, corresponds to the number of desired experts')

parser.add_argument('--append', '-apd', 
    default='default',
    help='strings to append to models to distinguish')

opt = parser.parse_args()

if opt.clusters < 0:
    from room_dataset import RoomDataset
    trainset = RoomDataset("training")
    model_g = Gating(trainset.num_experts)
    gating_loss = nn.NLLLoss()

else:
    from cluster_dataset import ClusterDataset
    trainset = ClusterDataset("training", num_clusters=opt.clusters)
    trainset.gating_probs = trainset.gating_probs.cuda()
    model_g = Gating(trainset.num_experts, capacity=2)
    gating_loss = nn.KLDivLoss()

trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=6)

model_file = 'gating_%s.net' % opt.append
model_g.cuda()
model_g.train()
optimizer_g = optim.Adam(model_g.parameters(), lr=opt.learningrate)

iteration = 0
epochs = math.ceil(opt.iterations / len(trainset))

train_log = open('log_init_gating_%s.txt' % (opt.append), 'w', 1)

for epoch in range(epochs):	
    print("----------------- Epoch: %d -----------------" % epoch)

    torch.save(model_g.state_dict(), model_file)
    print('Model saved.')

    for idx, image, focallength, gt_pose, gt_coords, gt_expert in trainset_loader:
        start_time = time.time()
        gt_expert = gt_expert.cuda()
        image = image.cuda()

        padX, padY, image = util.random_shift(image, Expert.OUTPUT_SUBSAMPLE / 2)

        gating = model_g(image)

        if opt.clusters < 0:
            loss = gating_loss(gating, gt_expert)
        else:
            loss = gating_loss(gating, trainset.gating_probs[idx])

        loss.backward()
        optimizer_g.step()
        optimizer_g.zero_grad()

        print('---------- Iteration: %6d, gating loss: %.2f, time: %.2fs ---------------\n' % (iteration, loss, time.time() - start_time), flush=True)
        train_log.write('%d %f \n' % (iteration, loss))

        iteration = iteration + 1

train_log.close()
print('Gating training finished.')
