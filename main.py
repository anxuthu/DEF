import os
import sys
import shutil
import time
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import optimizers
import all_reducer
from utils import *


def main():
    args = GetArgs()
    print(args, flush=True)

    ngpus_per_node = torch.cuda.device_count()
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    args.world_size = ngpus_per_node * args.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    cudnn.benchmark = True

    print("Use GPU: {} for training".format(args.gpu))

    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.dataset in ['cifar10', 'svhn']:
        import models
        model = models.__dict__[args.arch](num_classes=10)
    elif args.dataset == 'cifar100':
        import models
        model = models.__dict__[args.arch](num_classes=100)
    elif args.dataset == 'imagenet':
        import torchvision.models as models
        model = models.__dict__[args.arch]()
    elif args.dataset == 'tiny-imagenet':
        import models
        model = models.__dict__[args.arch](num_classes=200)
    else:
        assert False

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)

    for k, v in model.state_dict().items():
        dist.broadcast(v.data, src=args.root)

    # create datasets
    #args.batch_size = int(args.batch_size / args.world_size)
    train_dataset, val_dataset = GetDataset(args.dataset, args.path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True,
        seed=args.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    #reducer = all_reducer.RankK(
    #    random_seed=args.seed, device=args.gpu,
    #    reuse_query=args.reuse_query, rank=args.prank, #RankK
    #)
    reducer = all_reducer.URSB(
        random_seed=args.seed, device=args.gpu,
        compression=1.0 / args.ratio, #URSB
    )
    reducer2 = all_reducer.URSB(#TODO
        random_seed=args.seed, device=args.gpu,
        compression=1.0 / args.ratio2, #URSB
    )

    optimizer = optimizers.__dict__[args.optim](
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, period=args.period, period2=args.period2,
        reducer=reducer, reducer2=reducer2, coeff=args.coeff)

    # train and eval
    global best_acc1
    best_acc1 = 0
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train_eval(train_loader, val_loader, model, criterion, optimizer, epoch, args)


def train_eval(train_loader, val_loader, model, criterion, optimizer, epoch, args):
    epoch_time = AverageMeter('Time', ':9.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        args.epochs, [epoch_time, data_time, losses, top1, top5],
        prefix='[Worker {}] Epoch: '.format(args.rank))

    # switch to train mode
    model.train()

    end = time.time()
    for batch_idx, (images, target) in enumerate(train_loader):
        # adjust lr
        cur_step = batch_idx + epoch * len(train_loader)
        if cur_step % args.period == 0:
            lr = adjust_learning_rate(
                optimizer, epoch, batch_idx, len(train_loader), args)
            if batch_idx - args.period < 0:
                print('[Worker {}] Current lr: {:.6f} Epoch batches: {}'.format(
                    args.rank, lr, len(train_loader)), flush=True)

        # measure data loading time
        data_time.update(time.time() - end)
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        optimizer.zero_grad()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = Accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and update parameters
        loss.backward()
        optimizer.step()

        if (cur_step + 1) % args.period == 0:
            #for p in model.buffers():
            #    if len(p.shape) > 0:
            #        dist.all_reduce(p)
            #        p.div_(args.world_size)
            vec = parameters_to_vector(model.buffers())
            dist.all_reduce(vec)
            vec.div_(dist.get_world_size())
            vector_to_parameters(vec, model.buffers())

        epoch_time.update((time.time() - end)*len(train_loader))

        if (cur_step + 1) % args.period == 0:
            if val_loader and batch_idx + args.period >= len(train_loader):
                try:
                    optimizer.swap()
                except:
                    pass

                acc1 = Validate(val_loader, model, criterion, epoch, args.epochs,
                                args.rank, args.gpu)

                try:
                    optimizer.swap()
                except:
                    pass

                global best_acc1
                if acc1 > best_acc1:
                    best_acc1 = acc1
                    print('[Worker {}] Best Acc@1: {:6.2f}'.format(
                        args.rank, best_acc1, flush=True))
                model.train()

        end = time.time()

    progress.display(epoch)


def adjust_learning_rate(optimizer, epoch, batch_idx, n_batch, args):
    lr = args.lr
    if epoch < args.warm_up:
        lr = lr / args.warm_up / n_batch * (epoch * n_batch + batch_idx + 1)
    elif args.lr_schedule == 'const':
        for ds in args.decay_schedule:
            if epoch >= int(ds * args.epochs):
                lr *= 0.1
    elif args.lr_schedule == 'cos':
        cur_batch = (epoch - args.warm_up) * n_batch + batch_idx
        total_batch = (args.epochs - args.warm_up) * n_batch
        lr *= 0.5 * (1 + math.cos(math.pi * cur_batch / total_batch))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    main()
