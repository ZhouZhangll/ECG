'''
Training script for ecg classification
'''

from __future__ import print_function

import os
import cv2
import json
import time
import torch
import random
import shutil
import argparse
import numpy as np
import torch.nn as nn
import models as models
import torch.nn.parallel
import torch.optim as optim
import sklearn.metrics as skm
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import pandas as pd
from PIL import Image

from tqdm import tqdm
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
import  warnings
warnings.filterwarnings("ignore")


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ECG CNN Training')
# Datasets
parser.add_argument('-dt', '--dataset', default='ecg', type=str)
parser.add_argument('-ft', '--transformation', default=None, type=str)
parser.add_argument('-d', '--data', default='./data/ecg_data/image_datasets', type=str)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
# Optimization options
parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=4, type=int, metavar='N', help='train batchsize')
parser.add_argument('--test-batch', default=4, type=int, metavar='N', help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float, metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[15], help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='./checkpoint', type=str, metavar='PATH',help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

# Architecture
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', default=False,
                    help='evaluate model on validation set')

# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'ecg', 'Dataset can only be ecg.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


class Ecg_loader(Dataset):
    def __init__(self, ecg_dir, transform=None):
        """
        Args:
            ecg_dir (str): ECG数据根目录（包含metadata.csv和病例文件夹）
            transform (callable, optional): 数据增强函数
        """
        metadata_path = os.path.join(ecg_dir, "metadata.csv")
        self.idx2name = {'N': 0, 'A': 1}
        self.metadata = pd.read_csv(metadata_path)
        self.ecg_dir = ecg_dir
        self.transform = transform

        # 定义标准12导联顺序
        self.lead_order = [
            'I', 'II', 'III',
            'aVR', 'aVL', 'aVF',
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
        ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 读取元数据
        row = self.metadata.iloc[idx]
        folder_name = row["filename"]
        label_str = row["label"]
        label = self.idx2name[label_str]

        # 构建ECG文件夹路径
        folder_path = os.path.join(self.ecg_dir, folder_name)

        # 初始化存储12导联数据的列表
        ecg_data = []

        # 按标准顺序加载每个导联的图片
        for lead in self.lead_order:
            img_path = os.path.join(folder_path, f"{lead}.png")

            # 使用PIL加载并预处理图片
            with Image.open(img_path) as img:
                # 转换为灰度图并归一化到[0, 1]
                img_gray = img.convert('L')
                img_array = np.array(img_gray, dtype=np.float32) / 255.0

                # 添加额外维度 (H, W) -> (1, H, W)
                ecg_data.append(img_array)

        # 合并所有导联数据 (12, H, W)
        ecg = np.stack(ecg_data, axis=0)

        # 数据增强
        if self.transform:
            ecg = self.transform(ecg)

        # 转换为Tensor
        ecg_tensor = torch.from_numpy(ecg).float()  # (12, H, W)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return ecg_tensor, label_tensor


def evaluate(outputs, labels, label_names=None):
    gt = torch.cat(labels, dim=0)
    pred = torch.cat(outputs, dim=0)
    pred = torch.argmax(pred, dim=1)
    acc = torch.div(100*torch.sum((gt == pred).float()), gt.shape[0])
    print('accuracy :', acc)

    gt = gt.cpu().tolist()
    pred = pred.cpu().tolist()

    report = skm.classification_report(
        gt, pred,
        target_names=label_names,
        digits=3)
    scores = skm.precision_recall_fscore_support(
        gt,
        pred,
        average=None)
    print(report)
    print("F1 Average {:3f}".format(np.mean(scores[2][:3])))
    # print(scores)

    return True


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset %s' % args.dataset)

    dataloader = Ecg_loader
    train_path = args.data

    traindir = os.path.join(train_path, 'train')
    valdir = os.path.join(train_path, 'val')

    trainset = dataloader(traindir, transform=args.transformation)
    testset = dataloader(valdir, transform=args.transformation)

    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    idx2name = testset.idx2name
    label_names = idx2name.keys()

    num_classes = len(label_names)

    # Model

    model = models.__dict__['cnn_lstm_transformer_image'](num_classes=num_classes, input_channels = 12)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'ecg-cnn-lstm-transformer'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc= test(testloader, model, criterion, start_epoch, use_cuda, label_names=label_names)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))

        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda, label_names=label_names)

        # append logger file
        logger.append([state['lr'], train_loss.cpu(), test_loss.cpu(), train_acc.cpu(), test_acc.cpu()])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.jpg'))

    print('Best acc:')
    print(best_acc)


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # gt = []
    # pred = []

    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        # measure data loading time
        data_time.update(time.time() - end)

        # print(inputs.shape,targets.shape)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1= accuracy(outputs.data, targets.data, topk=(1,))
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1[0], inputs.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress

    # evaluate(pred, gt)
    return (losses.avg, top1.avg)


def test(testloader, model, criterion, epoch, use_cuda, label_names=None):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    gt = []
    pred = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        # compute output
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        gt.append(targets.data)
        pred.append(outputs.data)
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))


        losses.update(loss.data, inputs.size(0))
        top1.update(prec1[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress

    evaluate(pred, gt, label_names=label_names)
    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()
