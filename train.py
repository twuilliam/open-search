import argparse
import os
import json
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from data import create_splits, create_fewshot_splits, create_shape_splits
from data import create_multi_splits
from data import DataLoader, get_proxies
from models import LinearProjection, ConvNet
from models import ProxyNet, ProxyLoss
from utils import AverageMeter, to_numpy
from utils import get_semantic_fname, get_backbone, random_transform
from validate import extract_predict


# Training settings
parser = argparse.ArgumentParser(description='PyTorch SBIR')
# hyper-parameters
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--factor_lower', type=float, default=.1,
                    help='multiplicative factor of the LR for lower layers')
parser.add_argument('--seed', type=int, default=456, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--temperature', type=float, default=1., metavar='M',
                    help='temperature (default: 1.)')
parser.add_argument('--wd', type=float, default=5e-5, metavar='M',
                    help='weight decay (default: 5e-5)')
# flags
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--multi-gpu', action='store_true', default=False,
                    help='enables multi gpu training')
# model
parser.add_argument('--dim_embed', type=int, default=300, metavar='N',
                    help='how many dimensions in embedding (default: 300)')
parser.add_argument('--da', action='store_true', default=False,
                    help='data augmentation')
parser.add_argument('--backbone', type=str, default='resnet',
                    help='vgg16|vgg19|resnet|seresnet')
parser.add_argument('--word', type=str, default='word2vec',
                    help='Semantic space')
# setup
parser.add_argument('--fewshot', action='store_true', default=False,
                    help='few-shot experiment')
parser.add_argument('--gzsl', action='store_true', default=False,
                    help='Generalized setting, only works for Sketchy and TUB')
parser.add_argument('--shape', action='store_true', default=False,
                    help='3D shape recognition')
parser.add_argument('--mode', type=str, required=True,
                    help='im|sk')
# plumbing
parser.add_argument('--dataset', type=str, required=True,
                    help='Sketchy|TU-Berlin|SHREC13|SHREC14|PART-SHREC14|domainnet')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='zero-shot experiment')
parser.add_argument('--data_dir', type=str, metavar='DD',
                    default='data',
                    help='data folder path')
parser.add_argument('--exp_dir', type=str, default='exp', metavar='ED',
                    help='folder for saving exp')
parser.add_argument('--m', type=str, default='SBIR', metavar='M',
                    help='message')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

# get data splits
if 'domainnet' in args.dataset:
    args.domain = '_'.join(args.dataset.split('_')[1:])
    args.dataset = 'domainnet'
df_dir = os.path.join('aux', 'data', args.dataset)

if args.fewshot:
    splits = create_fewshot_splits(df_dir)
    df_train = splits[args.mode]['train']
    df_gal = splits[args.mode]['test']
elif args.shape:
    splits = create_shape_splits(df_dir)
    df_train = splits[args.mode]['train']
    df_gal = splits[args.mode]['test']
elif args.dataset in ['domainnet']:
    splits = create_multi_splits(
        df_dir, domain=args.domain, overwrite=args.overwrite)
    df_train = splits[args.mode]['train']
    df_gal = splits[args.mode]['test']
else:
    splits = create_splits(df_dir, args.overwrite, args.gzsl)
    df_train = splits[args.mode]['train']
    df_gal = splits[args.mode]['gal']

args.n_classes = len(df_train['cat'].cat.categories)
args.n_classes_gal = len(df_gal['cat'].cat.categories)

# create experiment folder
dname = '%s_%s' % (args.dataset, args.mode)
if args.dataset == 'domainnet' and args.mode == 'im':
    dname = dname + '_' + args.domain
path = os.path.join(args.exp_dir, dname)
if not os.path.exists(path):
    os.makedirs(path)

# saving logs
with open(os.path.join(path, 'config.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=4, sort_keys=True)

with open(os.path.join(path, 'logs.txt'), 'w') as f:
    f.write('Experiment with SBIR\n')


def write_logs(txt, logpath=os.path.join(path, 'logs.txt')):
    with open(logpath, 'a') as f:
        f.write('\n')
        f.write(txt)


def main():
    # data normalization
    input_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # data loaders
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

    if args.da:
        train_transforms = transforms.Compose([
            random_transform,
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            normalize])
    else:
        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize])

    train_loader = torch.utils.data.DataLoader(
        DataLoader(df_train, train_transforms,
                   root=args.data_dir, mode=args.mode),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        DataLoader(df_gal, test_transforms,
                   root=args.data_dir, mode=args.mode),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    # instanciate the models
    output_shape, backbone = get_backbone(args)
    embed = LinearProjection(output_shape, args.dim_embed)
    model = ConvNet(backbone, embed)

    # instanciate the proxies
    fsem = get_semantic_fname(args.word)
    path_semantic = os.path.join('aux', 'Semantic', args.dataset, fsem)
    train_proxies = get_proxies(
        path_semantic, df_train['cat'].cat.categories)
    test_proxies = get_proxies(
        path_semantic, df_gal['cat'].cat.categories)

    train_proxynet = ProxyNet(args.n_classes, args.dim_embed,
                              proxies=torch.from_numpy(train_proxies))
    test_proxynet = ProxyNet(args.n_classes_gal, args.dim_embed,
                             proxies=torch.from_numpy(test_proxies))

    # criterion
    criterion = ProxyLoss(args.temperature)

    if args.multi_gpu:
        model = nn.DataParallel(model)

    if args.cuda:
        backbone.cuda()
        embed.cuda()
        model.cuda()
        train_proxynet.cuda()
        test_proxynet.cuda()

    parameters_set = []

    low_layers = []
    upper_layers = []

    for c in backbone.children():
        low_layers.extend(list(c.parameters()))
    for c in embed.children():
        upper_layers.extend(list(c.parameters()))

    parameters_set.append({'params': low_layers,
                           'lr': args.lr * args.factor_lower})
    parameters_set.append({'params': upper_layers,
                           'lr': args.lr * 1.})

    optimizer = optim.SGD(
        parameters_set, lr=args.lr,
        momentum=0.9, nesterov=True,
        weight_decay=args.wd)

    n_parameters = sum([p.data.nelement()
                        for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    scheduler = CosineAnnealingLR(
        optimizer, args.epochs * len(train_loader), eta_min=3e-6)

    print('Starting training...')
    for epoch in range(args.start_epoch, args.epochs + 1):
        # update learning rate
        scheduler.step()

        # train for one epoch
        train(train_loader, model,
              train_proxynet.proxies.weight, criterion,
              optimizer, epoch, scheduler)

        val_acc = evaluate(
            test_loader, model,
            test_proxynet.proxies.weight, criterion)

        # saving
        if epoch == args.epochs:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict()})

    print('\nResults on test set (end of training)')
    write_logs('\nResults on test set (end of training)')
    test_acc = evaluate(
        test_loader, model,
        test_proxynet.proxies.weight, criterion)


def train(train_loader, model,
          proxies, criterion, optimizer, epoch, scheduler):
    """Training loop for one epoch"""
    batch_time = AverageMeter()
    data_time = AverageMeter()

    val_loss = AverageMeter()
    val_acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (x, y) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if len(x.shape) == 5:
            batch_size, nviews = x.shape[0], x.shape[1]
            x = x.view(batch_size * nviews, 3, 224, 224)

        if len(y) == args.batch_size:
            if args.cuda:
                x = x.cuda()
                y = y.cuda()

            x = Variable(x)

            # embed
            x_emb = model(x)

            loss, acc = criterion(x_emb, y, proxies)

            val_loss.update(to_numpy(loss), x.size(0))
            val_acc.update(acc, x.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    txt = ('Epoch [%d] (Time %.2f Data %.2f):\t'
           'Loss %.4f\t Acc %.4f' %
           (epoch, batch_time.avg * i, data_time.avg * i,
            val_loss.avg, val_acc.avg * 100.))
    print(txt)
    write_logs(txt)


def evaluate(loader, model, proxies, criterion):
    x, y, acc = extract_predict(loader, model, proxies, criterion)
    txt = ('.. Acc: %.02f\t' % (acc * 100))
    print(txt)
    write_logs(txt)
    return acc


def save_checkpoint(state, folder=path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(path, filename))


if __name__ == '__main__':
    main()
