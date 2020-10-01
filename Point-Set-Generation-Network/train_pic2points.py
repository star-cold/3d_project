from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets import PartDataset
import torch.nn.functional as F
import torch.cuda as cuda
from pic2points_model import pic2points
from torch.nn.parallel import DataParallel
from torch.autograd import Variable
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--model', type=str, default='',  help='model path')

def batch_pairwise_dist(x, y):
    # 32, 2500, 3
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P


def batch_NN_loss(x, y):
    bs, num_points, points_dim = x.size()
    dist1 = torch.sqrt(batch_pairwise_dist(x, y))
    values1, indices1 = dist1.min(dim=2)

    dist2 = torch.sqrt(batch_pairwise_dist(y, x))
    values2, indices2 = dist2.min(dim=2)
    a = torch.div(torch.sum(values1,1), num_points)
    b = torch.div(torch.sum(values2,1), num_points)
    sum = torch.div(torch.sum(a), bs) + torch.div(torch.sum(b), bs)

    return sum


def main():
    global args
    opt = parser.parse_args()
    print (opt)

    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', pic2point = True, npoints = opt.num_points)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))
    print("number of training data:"+ str(len(dataset)))
    test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', pic2point = True, train = False, npoints = opt.num_points)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,shuffle=True, num_workers=int(opt.workers))
    print("number of training data:"+ str(len(test_dataset)))

    # creat model
    print("model building...")
    model = pic2points(num_points=opt.num_points)
    model.cuda()

    # load pre-existing weights
    if opt.model != '':
        model.load_state_dict(torch.load(opt.model))

    optimizer = torch.optim.Adadelta(model.parameters(), weight_decay=1e-4)
    num_batch = len(dataset) / opt.batchSize

    print('training mode ------------------')
    for epoch in range(opt.nepoch):
        print("epoch:"+str(epoch))
        for i, data in enumerate(dataloader, 0):
            im, points = data
            im, points = Variable(im), Variable(points)
            im, points = im.cuda(), points.cuda()
            pred = model(im)
            loss = batch_NN_loss(pred, points).cuda()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 is 0:
                print("training loss is:" + str(loss.data[0]))

        loss_test = 0
        for i, data in enumerate(dataloader, 0):
            im_test, points_test = data
            im_test, points_test = Variable(im_test), Variable(points_test)
            im_test, points_test = im_test.cuda(), points_test.cuda()
            pred_test = model(im_test)
            loss_test = batch_NN_loss(pred_test, points_test).cuda()
        print("Testing loss is:" + str(loss_test.data[0]))

if __name__ == '__main__':
    num_cuda = cuda.device_count()
    print("number of GPUs have been detected:" + str(num_cuda))
    with torch.cuda.device(1):
        main()

