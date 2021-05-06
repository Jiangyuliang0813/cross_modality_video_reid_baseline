import os
import sys
import time


import os.path as osp
import numpy as np
import argparse


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler

import transforms as T
from video_loader import VideoDataset_train, VideoDataset_test

from samplers import IdentitySampler
from data_manager import CUHK60
from eval_metrics import evaluate

from model import binetwork
from loss import OriTripletLoss, TripletLoss_WRT
from utils import AverageMeter, Logger, save_checkpoint, GenIdx

cuda_devices = '0'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_dir = './/'
is_evaluate = False
seed = 12

height = 224
width = 112
seq_lenth = 4
train_batch = 2
workers = 4
num_instances = 4

lr = 0.0003
w_decay = 5e-04
step_size = 200
gamma = 0.1
start_epoch = 0
max_epoch = 120
print_freq = 2
stepsize = 200
# 等下要改
margin, lambda_intra = 3, 4
# test
test_batch = 1 # b =1 后面
num_pos = 4
eval_step = 1


test_mode = [1, 2]  # rgb,ir


def main():
    torch.manual_seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
    use_gpu = torch.cuda.is_available()
    # print('我执行到58行')

    pin_memory = True if use_gpu else False

    # if not is_evaluate:
    #     sys.stdout = Logger(osp.join(save_dir, 'log_train.txt'))
    # else:
    #     sys.stdout = Logger(osp.join(save_dir, 'log_test.txt'))

    if use_gpu:
        print("Currently using GPU {}".format(cuda_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format('CUHK60'))
    dataset = CUHK60()

    # 或许可加reid通用数据增强
    transform_train = T.Compose([
        T.Random2DTranslation(height, width),
        T.RandomHorizontalFlip(),

        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 配合sampler，获取dataset中的rgb、ir中id图片位置信息
    rgb_pos, ir_pos = GenIdx(dataset.rgb_label, dataset.ir_label)

    sampler = IdentitySampler(dataset.ir_label, dataset.rgb_label, rgb_pos, ir_pos, num_pos, train_batch)
    # for x in sampler:
    #     print(x)

    trainloader = DataLoader(
        VideoDataset_train(dataset.train_ir, dataset.train_rgb, seq_len=4, sample='random', transform=transform_train),
        sampler=sampler,
        batch_size=train_batch, num_workers=workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        VideoDataset_test(dataset.query, seq_len=4, sample='dense', transform=transform_test),
        batch_size=test_batch, shuffle=False, num_workers=workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        VideoDataset_test(dataset.gallery, seq_len=4, sample='dense', transform=transform_test),
        batch_size=test_batch, shuffle=False, num_workers=workers,
        pin_memory=pin_memory, drop_last=False,
    )

    # 等下要改--已经改完一遍--第二遍加入data
    model = binetwork(dataset.num_train_pids)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    # 等下要改loss中的东西-已完成
    criterion_idloss = nn.CrossEntropyLoss()
    criterion_rankingloss = TripletLoss_WRT()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)

    if stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma)

        if use_gpu:
            model = nn.DataParallel(model).cuda()

        start_time = time.time()
        best_rank1 = -np.inf
        for epoch in range(start_epoch, max_epoch):
            print("==> Epoch {}/{}".format(epoch + 1, max_epoch))
            train(model, criterion_idloss, criterion_rankingloss, optimizer, trainloader, use_gpu)

            if stepsize > 0: scheduler.step()

            if eval_step > 0 and (epoch + 1) % eval_step == 0 or (epoch + 1) == max_epoch:
                print("==>Test")
                rank1 = test(model, queryloader, galleryloader, use_gpu)
                is_best = rank1 > best_rank1
                if is_best: best_rank1 = rank1

                if use_gpu:
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                save_checkpoint({
                    'state_dict': state_dict,
                    'rank1': rank1,
                    'epoch': epoch,
                }, is_best, osp.join(save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    elapsed = round(time.time() - start_time)

    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(model, criterion_idloss, criterion_rankingloss, optimizer, trainloader, use_gpu):
    model.train()
    losses = AverageMeter()
    correct = 0

    for batch_idx, (imgs_ir, pids_ir, camid_ir, imgs_rgb, pids_rgb, camid_rgb) in enumerate(trainloader):
        # 同时读取ir,rgb输入
        # print('------------------------')
        # print('batch_idx = {}'.format(batch_idx))
        # print('img_rgb_id = {}'.format(pids_rgb))
        # print('img_ir_id = {}'.format(pids_ir))
        # print('imgs_ir = {}'.format(imgs_ir.shape))
        # print('imgs_rgb = {}'.format(imgs_rgb.shape))
        pids = torch.cat((pids_rgb, pids_ir), 0)
        if use_gpu:
            imgs_rgb, imgs_ir, pids = imgs_rgb.cuda(), imgs_ir.cuda(), pids.cuda()

        imgs_rgb, imgs_ir, pids = Variable(imgs_rgb), Variable(imgs_ir), Variable(pids)

        feat, pred0 = model(imgs_rgb, imgs_ir)

        ranking_loss, prec = criterion_rankingloss(feat, pids)
        identity_loss = criterion_idloss(pred0, pids)

        correct += (prec / 2)
        _, predicted = pred0.max(1)
        correct += (predicted.eq(pids).sum().item() / 2)

        loss = ranking_loss + identity_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), pids.size(0))

        if (batch_idx + 1) % print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx + 1, len(trainloader), losses.val, losses.avg))


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    model.eval()

    qf, q_pids, q_camids = [], [], []
    # ir -> rgb
    with torch.no_grad():
        # 读取query集合中的图片数据，并将其编码
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            imgs = Variable(imgs)
            # print(imgs.size())
            b, n, s, c, h, w = imgs.size()

            assert (b == 1)

            imgs = imgs.view(b * n, s, c, h, w)
            # print('size changed to {}'.format(imgs.size()))
            features = model(imgs, imgs, test_mode[1])
            features = features.view(n, -1)
            features = torch.mean(features, 0)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
    qf = torch.stack(qf)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

    gf, g_pids, g_camids = [], [], []
    with torch.no_grad():
        # 读取gallery集中的数据，进行编码
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()
            imgs = Variable(imgs)
            b, n, s, c, h, w = imgs.size()
            imgs = imgs.view(b * n, s, c, h, w)
            assert (b == 1)

            features = model(imgs, imgs, test_mode[0])
            features = features.view(n, -1)

            features, _ = torch.max(features, 0)
            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))
    print("Computing distance matrix")

    m, n = qf.size(0), gf.size(0)
    # 制作距离矩阵
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    # print("dismat = {}".format(distmat.shape))
    # print("q_pids = {}".format(q_pids))
    # print("g_pids = {}".format(g_pids))
    # print("q_camids = {}".format(q_camids))
    # print("g_camids = {}".format(g_camids))

    print("Computing CMC and mAP")

    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    return cmc[0]


if __name__ == '__main__':
    main()
