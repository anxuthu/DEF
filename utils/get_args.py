import argparse

def GetArgs():
    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('--path', default='/export/Data/cifar', type=str, help='path to dataset')
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('-a', '--arch', default='vgg16bn', help='model architecture')
    parser.add_argument('-j', '--workers', default=4, type=int, help='data loading workers')
    parser.add_argument('--epochs', default=300, type=int, help='total epochs')
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('-eb', '--eval-batch-size', default=128, type=int)
    parser.add_argument('--world-size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int, help='node rank')
    parser.add_argument('--root', default=0, type=int, help='root node')
    parser.add_argument('--seed', default=0, type=int, help='dist sampler')
    parser.add_argument('--dist-url', default='tcp://localhost:23450', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='NCCL', type=str, help='dist backend')

    ### optimizer hyperparameters ###
    parser.add_argument('-o', '--optim', default='SGD', type=str)
    parser.add_argument('-wd', '--weight-decay', default=5e-4, type=float)
    parser.add_argument('-m', '--momentum', default=0.9, type=float)
    parser.add_argument('-p', '--period', default=1, type=int, help='local steps')
    parser.add_argument('-p2', '--period2', default=None, type=int, help='saef')
    parser.add_argument('--coeff', default=1, type=float)

    ### learning rate ###
    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('-wp', '--warm-up', default=5, type=int, help='lr warm-up')
    parser.add_argument('-ls', '--lr-schedule', default='cos', type=str, choices=['const', 'cos'])
    parser.add_argument('-ds', '--decay-schedule', type=float, nargs='+', default=[0.5, 0.75], help='lr decaying epochs')

    ### compression ###
    #parser.add_argument('--reducer', default='RankK', choices=['RankK', 'URSB', 'TopK'])
    parser.add_argument('--prank', default=1, type=int, help='PowerSGD rank')
    parser.add_argument('-rq', '--reuse-query', action='store_true')
    parser.add_argument('--ratio', default=1, type=float, help='URSB')

    return parser.parse_args()
