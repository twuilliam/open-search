import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from data import create_fewshot_splits
from data import DataLoader, get_proxies
from models import LinearProjection, ConvNet
from models import ProxyNet, ProxyLoss
from utils import get_semantic_fname, get_backbone
from validate import extract_predict, retrieve


# Training settings
parser = argparse.ArgumentParser(description='PyTorch SBIR')
parser.add_argument('--im-path', type=str, default='exp', metavar='ED',
                    help='im model path')
parser.add_argument('--sk-path', type=str, default='exp', metavar='ED',
                    help='sk model path')
parser.add_argument('--rewrite', action='store_true', default=False,
                    help='Do not consider existing saved features')
parser.add_argument('--mixing', action='store_true', default=False,
                    help='Mix w2v with sk representations')
parser.add_argument('--seed', type=int, default=1234,
                    help='Seed for selecting the sketches')


args = parser.parse_args()
im_path = os.path.dirname(args.im_path)
SEED = args.seed
GROUPS = 500

with open(os.path.join(im_path, 'config.json')) as f:
    tmp = json.load(f)

tmp['im_model_path'] = args.im_path
tmp['sk_model_path'] = args.sk_path
tmp['rewrite'] = args.rewrite
tmp['mixing'] = args.mixing
args = type('parser', (object,), tmp)

# get data splits
df_dir = os.path.join('aux', 'data', args.dataset)
splits = create_fewshot_splits(df_dir)


def main():
    # data normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # data loaders
    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
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

            df_gal = splits[domain]['test']
            fsem = get_semantic_fname(args.word)
            path_semantic = os.path.join('aux', 'Semantic', args.dataset, fsem)
            test_proxies = get_proxies(
                path_semantic, df_gal['cat'].cat.categories)
        else:
            df_gal = splits[domain]['test']

            test_loader = torch.utils.data.DataLoader(
                DataLoader(df_gal, test_transforms,
                           root=args.data_dir, mode=domain),
                batch_size=args.batch_size * 1, shuffle=False, **kwargs)

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

            np.savez(
                fpath,
                features=feats[domain], labels=labels[domain], acc=acc)

            txt = ('Domain (%s): Acc %.2f' % (domain, acc * 100.))
            print(txt)

    print('\nFew-Shot')
    fs(feats, labels, test_proxies)


def fs(feats, labels, proxies):

    feats['sk'] = L2norm(feats['sk'])
    feats['im'] = L2norm(feats['im'])
    proxies = L2norm(proxies)

    # word vectors
    acc = classify(feats['im'], labels['im'], proxies)
    print('word vectors:\t\t%.2f' % (acc * 100))

    # few-shot with sketches
    info = do_sk_fewshot(feats['sk'], labels['sk'],
                         feats['im'], labels['im'],
                         k=1, v=True)
    to_save(info, 'best_worst.npy')
    do_sk_fewshot(feats['sk'], labels['sk'], feats['im'], labels['im'], k=5)

    if args.mixing:
        do_sk_fewshot_mixer(feats['sk'], labels['sk'],
                            feats['im'], labels['im'],
                            proxies, k=1)

        do_sk_fewshot_mixer(feats['sk'], labels['sk'],
                            feats['im'], labels['im'],
                            proxies, k=5)

    # few-shot with images
    do_im_fewshot(feats['im'], labels['im'])
    do_im_fewshot(feats['im'], labels['im'], k=5)

    if args.mixing:
        info = do_im_fewshot_mixer(
            feats['im'], labels['im'], proxies, k=1, v=True)
        to_save(info, 'im_1_mixture.npy')
        info = do_im_fewshot_mixer(
            feats['im'], labels['im'], proxies, k=5, v=True)
        to_save(info, 'im_5_mixture.npy')


def do_sk_fewshot(query_x, query_y, gallery_x, gallery_y,
                  k=1, v=False):
    np.random.seed(SEED)

    best_acc = 0.
    worst_acc = 1.

    acc = []
    for i in range(GROUPS):
        new_p, idx = shot_selector(query_x, query_y, k=k)
        acc.append(classify(gallery_x, gallery_y, new_p))

        if acc[-1] > best_acc:
            best_acc = acc[-1]
            best_idx = list(idx)
        elif acc[-1] < worst_acc:
            worst_acc = acc[-1]
            worst_idx = list(idx)

    print('%d-shot (sketches):\t%.2f (+/- %.2f)' %
          (k, np.mean(acc) * 100, np.std(acc) * 100))

    if v:
        return parse_samples((best_acc, best_idx), (worst_acc, worst_idx))


def parse_samples(best, worst):
    info = {}

    info['best'] = {}
    info['best']['acc'] = best[0]
    info['best']['idx'] = best[1]

    info['worst'] = {}
    info['worst']['acc'] = worst[0]
    info['worst']['idx'] = worst[1]

    return info


def to_save(info, fname):
    dirname = os.path.dirname(args.__dict__['sk_model_path'])
    fpath = os.path.join(dirname, fname)
    np.save(fpath, info)


def do_sk_fewshot_mixer(query_x, query_y, gallery_x, gallery_y, proxies, k=1):
    np.random.seed(SEED)
    acc = []
    alpha = 0.7
    for i in range(GROUPS):
        new_p, _ = shot_selector_mixer(
            query_x, query_y, proxies, k=k, alpha=alpha)
        acc.append(classify(gallery_x, gallery_y, new_p))
    print('(w. refinement) %d-shot (sketches):\t%.2f (+/- %.2f)' %
          (k, np.mean(acc) * 100, np.std(acc) * 100))


def do_im_fewshot(gallery_x, gallery_y, k=1):
    np.random.seed(SEED)
    acc = []
    for i in range(GROUPS):
        new_p, idx = shot_selector(gallery_x, gallery_y, k=k)
        cond = np.isin(np.arange(len(gallery_y)), idx)
        acc.append(classify(gallery_x[~cond], gallery_y[~cond], new_p))
    print('%d-shot (images):\t%.2f (+/- %.2f)' %
          (k, np.mean(acc) * 100, np.std(acc) * 100))


def do_im_fewshot_mixer(gallery_x, gallery_y, proxies, k=1, v=False):
    np.random.seed(SEED)
    acc = []
    alpha = 0.7
    for i in range(GROUPS):
        new_p, idx = shot_selector_mixer(
            gallery_x, gallery_y, proxies, k=k, alpha=alpha)
        cond = np.isin(np.arange(len(gallery_y)), idx)
        acc.append(classify(gallery_x[~cond], gallery_y[~cond], new_p))
    print('(w. refinement) %d-shot (images):\t%.2f (+/- %.2f)' %
          (k, np.mean(acc) * 100, np.std(acc) * 100))


def classify(feats, labels, proxies):
    idx = retrieve(feats, proxies)
    acc = np.mean(idx[:, 0] == labels)
    return acc


def shot_selector(feats, labels, k=1):
    vv = np.unique(labels)
    proxies = []
    all_idx = []
    for v in vv:
        to_select = np.argwhere(labels == v).squeeze()
        idx = np.random.choice(to_select, size=k, replace=False)
        proxies.append(np.mean(feats[idx, :], axis=0))
        all_idx.extend(idx)
    return np.asarray(proxies), np.asarray(all_idx)


def shot_selector_mixer(feats, labels, proxies, k=1, alpha=0.5):
    vv = np.unique(labels)
    new_proxies = []
    all_idx = []
    for v in vv:
        to_select = np.argwhere(labels == v).squeeze()
        idx = np.random.choice(to_select, size=k, replace=False)
        if alpha == 0:
            new_proxies.append(np.mean(feats[idx, :], axis=0))
        elif alpha == 1:
            new_proxies.append(np.mean(proxies[v, :][None, :], axis=0))
        else:
            if k == 1:
                second = feats[idx, :]
            else:
                second = L2norm(np.mean(feats[idx, :], axis=0)[None, :])
            tmp = np.concatenate((alpha * proxies[v, :][None, :],
                                  (1 - alpha) * second))
            new_proxies.append(np.mean(tmp, axis=0))

        all_idx.extend(idx)
    return np.asarray(new_proxies), np.asarray(all_idx)


def L2norm(x):
    return x / np.linalg.norm(x, axis=1)[:, None]


def write_logs(txt, logpath):
    with open(logpath, 'a') as f:
        f.write('\n')
        f.write(txt)


if __name__ == '__main__':
    main()
