import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from data import create_splits, create_shape_splits, create_multi_splits
from data import DataLoader, get_proxies
from models import LinearProjection, ConvNet
from models import ProxyNet, ProxyLoss
from utils import get_semantic_fname, get_backbone
from validate import extract_predict
from validate import L2norm
from validate import retrieve, KNN, score, score_shape


# Training settings
parser = argparse.ArgumentParser(description='PyTorch SBIR')
parser.add_argument('--im-path', type=str, default='exp', metavar='ED',
                    help='im model path')
parser.add_argument('--sk-path', type=str, default='exp', metavar='ED',
                    help='sk model path')
parser.add_argument('--new-data-path', type=str, default='', metavar='ED',
                    help='overwrite the original data path')
parser.add_argument('--rewrite', action='store_true', default=False,
                    help='Do not consider existing saved features')
parser.add_argument('--train', action='store_true', default=False,
                    help='Also extract the training set')


args = parser.parse_args()
im_path = os.path.dirname(args.im_path)

with open(os.path.join(im_path, 'config.json')) as f:
    tmp = json.load(f)

tmp['im_model_path'] = args.im_path
tmp['sk_model_path'] = args.sk_path
tmp['rewrite'] = args.rewrite
tmp['train'] = args.train
tmp['new_data_path'] = args.new_data_path
args = type('parser', (object,), tmp)

if not args.new_data_path == '':
    args.data_dir = args.new_data_path

# get data splits
df_dir = os.path.join('aux', 'data', args.dataset)
if args.shape:
    splits = create_shape_splits(df_dir)
elif args.dataset == 'domainnet':
    splits = create_multi_splits(
        df_dir, domain=args.domain, overwrite=args.overwrite)
else:
    splits = create_splits(df_dir, args.overwrite, args.gzsl)


def main():
    # data normalization
    input_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # data loaders
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize])

    feats = {}
    labels = {}

    for domain in ['im', 'sk']:
        key = '_'.join([domain, 'model_path'])
        dirname = os.path.dirname(args.__dict__[key])
        fpath = os.path.join(dirname, 'features.npz')

        results_path = os.path.join(dirname, 'results.txt')

        if os.path.isfile(fpath) and args.rewrite is False:
            data = np.load(fpath)
            feats[domain] = data['features']
            labels[domain] = data['labels']

            txt = ('Domain (%s): Acc %.2f' % (domain, data['acc'] * 100.))
            print(txt)
            write_logs(txt, results_path)

            df_gal = splits[domain]['gal']
            fsem = get_semantic_fname(args.word)
            path_semantic = os.path.join('aux', 'Semantic', args.dataset, fsem)
            test_proxies = get_proxies(
                path_semantic, df_gal['cat'].cat.categories)
        else:
            df_gal = splits[domain]['gal']

            test_loader = torch.utils.data.DataLoader(
                DataLoader(df_gal, test_transforms,
                           root=args.data_dir, mode=domain),
                batch_size=args.batch_size * 10, shuffle=False, **kwargs)

            # instanciate the models
            output_shape, backbone = get_backbone(args)
            embed = LinearProjection(output_shape, args.dim_embed)
            model = ConvNet(backbone, embed)

            # instanciate the proxies
            fsem = get_semantic_fname(args.word)
            path_semantic = os.path.join('aux', 'Semantic', args.dataset, fsem)
            test_proxies = get_proxies(
                path_semantic, df_gal['cat'].cat.categories)

            test_proxynet = ProxyNet(args.n_classes_gal, args.dim_embed,
                                     proxies=torch.from_numpy(test_proxies))

            # criterion
            criterion = ProxyLoss(args.temperature)

            if args.multi_gpu:
                model = nn.DataParallel(model)

            # loading
            checkpoint = torch.load(args.__dict__[key])
            model.load_state_dict(checkpoint['state_dict'])
            txt = ("\n=> loaded checkpoint '{}' (epoch {})"
                   .format(args.__dict__[key], checkpoint['epoch']))
            print(txt)
            write_logs(txt, results_path)

            if args.cuda:
                backbone.cuda()
                embed.cuda()
                model.cuda()
                test_proxynet.cuda()

            txt = 'Extracting testing set (%s)...' % (domain)
            print(txt)
            x, y, acc = extract_predict(
                test_loader, model,
                test_proxynet.proxies.weight, criterion)

            feats[domain] = x
            labels[domain] = y

            np.savez(fpath, features=feats[domain], labels=labels[domain], acc=acc)

            fpath_train = os.path.join(dirname, 'features_train.npz')
            if args.train and not os.path.isfile(fpath_train):
                df_train = splits[domain]['train']

                train_loader = torch.utils.data.DataLoader(
                    DataLoader(df_train, test_transforms,
                               root=args.data_dir, mode=domain),
                    batch_size=args.batch_size * 10, shuffle=False, **kwargs)

                train_proxies = get_proxies(
                    path_semantic, df_train['cat'].cat.categories)

                train_proxynet = ProxyNet(args.n_classes_gal, args.dim_embed,
                                          proxies=torch.from_numpy(train_proxies))
                train_proxynet.cuda()
                txt = 'Extracting training set (%s)...' % (domain)
                print(txt)

                x, y, _ = extract_predict(
                    train_loader, model,
                    train_proxynet.proxies.weight, criterion)

                fpath = os.path.join(dirname, 'features_train.npz')

                np.savez(
                    fpath,
                    features=feats[domain], features_train=x,
                    labels=labels[domain], labels_train=y,
                    acc=acc)

            txt = ('Domain (%s): Acc %.2f' % (domain, acc * 100.))
            print(txt)
            write_logs(txt, results_path)

    if args.shape:
        print('\nRetrieval per model')
        new_feat_im, new_labels_im = average_views(
            splits['im']['test'], feats['im'], labels['im'])

        idx = retrieve(feats['sk'], new_feat_im)

        metrics = score_shape(labels['sk'], new_labels_im, idx)
        names = ['NN', 'FT', 'ST', 'E', 'nDCG', 'mAP']
        txt = [('%s %.3f' % (name, value)) for name, value in zip(names, metrics)]
        txt = '\t'.join(txt)
        print(txt)
        write_logs(txt, results_path)

        print('\nRetrieval per model with refinement')

        alpha = 0.4

        g_sk_x = KNN(feats['sk'], new_feat_im, K=1, mode='ones')
        new_sk_x = slerp(alpha, L2norm(feats['sk']), L2norm(g_sk_x))
        idx = retrieve(new_sk_x, new_feat_im)
        metrics = score_shape(labels['sk'], new_labels_im, idx)
        names = ['NN', 'FT', 'ST', 'E', 'nDCG', 'mAP']
        txt = [('%s %.3f' % (name, value)) for name, value in zip(names, metrics)]
        txt = '\t'.join(txt)
        print(txt)
        write_logs(txt, results_path)

    else:
        print('\nRetrieval')
        txt = evaluate(feats['im'], labels['im'],
                       feats['sk'], labels['sk'])
        print(txt)
        write_logs(txt, results_path)

        print('\nRetrieval with refinement')
        if args.overwrite:
            alpha = 0.7
        else:
            alpha = 0.4

        g_sk_x = KNN(feats['sk'], feats['im'], K=1, mode='ones')

        new_sk_x = slerp(alpha, L2norm(feats['sk']), L2norm(g_sk_x))
        txt = evaluate(
            feats['im'], labels['im'],
            new_sk_x, labels['sk'])
        print(txt)
        write_logs(txt, results_path)


def evaluate(im_x, im_y, sk_x, sk_y, classnames=None):
    idx = retrieve(sk_x, im_x)
    prec, mAP = score(sk_y, im_y, idx)
    txt = ('mAP@all: %.04f Prec@100: %.04f\t' % (mAP, prec))
    return txt


def average_views(splits, x, y):
    ids = splits['id'].unique()
    feat = []
    labels = []
    for cad_id in ids:
        cond = splits['id'] == cad_id
        feat.append(np.mean(x[cond, :], axis=0))
        labels.append(y[cond][0])
    return np.asarray(feat), np.asarray(labels)


def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1."""
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif np.allclose(low, high):
        return low
    omega = np.arccos(np.einsum('ij, ij->i', low, high))
    so = np.sin(omega)
    return (np.sin((1.0-val)*omega) / so)[:, None] * low + (np.sin(val*omega)/so)[:, None] * high


def write_logs(txt, logpath):
    with open(logpath, 'a') as f:
        f.write('\n')
        f.write(txt)


if __name__ == '__main__':
    main()
