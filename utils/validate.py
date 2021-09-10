import torch
import time
from .meters import AverageMeter, ProgressMeter
from .accuracy import Accuracy

def Validate(val_loader, model, criterion, epoch, epochs, rank=0, gpu=None):
    epoch_time = AverageMeter('Time', ':9.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        epochs,
        [epoch_time, losses, top1, top5],
        prefix='[Worker {}] Test: '.format(rank))

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for batch_idx, (images, target) in enumerate(val_loader):
            if gpu is not None:
                images = images.cuda(gpu, non_blocking=True)
                target = target.cuda(gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = Accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            epoch_time.update((time.time() - end)*len(val_loader))
            end = time.time()

        progress.display(epoch)

    return top1.avg
