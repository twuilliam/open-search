import argparse
import os
import itertools
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as plt
from validate import L2norm
from validate import retrieve, KNN, score
from utils import heatmap, annotate_heatmap


# Training settings
parser = argparse.ArgumentParser(description='PyTorch SBIR')
parser.add_argument('--dir-path', type=str, default='exp', metavar='ED',
                    help='directory with domainnet models')

args = parser.parse_args()

GROUPS = 500
SEED = 1234


def get_config():
    # iterate over folders
    configs = {}
    for path in os.listdir(args.dir_path):
        fname = os.path.join(args.dir_path, path, 'config.json')
        if os.path.isfile(fname):

            with open(fname) as f:
                tmp = json.load(f)

            if tmp['mode'] == 'im':
                configs[tmp['domain']] = tmp
                configs[tmp['domain']]['working_path'] = os.path.join(
                    args.dir_path, path)
            else:
                configs['quickdraw'] = tmp
                configs['quickdraw']['working_path'] = os.path.join(
                    args.dir_path, path)
    return configs


def any2any_retrieval(configs):
    keys = configs.keys()
    keys.sort()

    res = {}
    for k in keys:
        res[k] = {}
        for j in keys:
            res[k][j] = 0.

    for (source, target) in itertools.combinations(keys, 2):
        feats = {}
        labels = {}

        for domain in [source, target]:
            dirname = configs[domain]['working_path']
            fpath = os.path.join(dirname, 'features.npz')

            data = np.load(fpath)

            feats[domain] = {}
            labels[domain] = {}

            feats[domain] = data['features']
            labels[domain] = data['labels']

        if res[source][source] == 0:
            print('\nRetrieval from %s to %s' % (source, source))
            tmp = cross_domain_retrieval(
                feats[source], labels[source],
                feats[source], labels[source],
                zeroshot=configs[source]['overwrite'])
            res[source][source] = tmp

        if res[target][target] == 0:
            print('\nRetrieval from %s to %s' % (target, target))
            tmp = cross_domain_retrieval(
                feats[target], labels[target],
                feats[target], labels[target],
                zeroshot=configs[source]['overwrite'])
            res[target][target] = tmp

        print('\nRetrieval from %s to %s' % (source, target))
        tmp = cross_domain_retrieval(
            feats[source], labels[source],
            feats[target], labels[target],
            zeroshot=configs[source]['overwrite'])
        res[source][target] = tmp

        print('\nRetrieval from %s to %s' % (target, source))
        tmp = cross_domain_retrieval(
            feats[target], labels[target],
            feats[source], labels[source],
            zeroshot=configs[source]['overwrite'])
        res[target][source] = tmp

    # col: source, row: target
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(args.dir_path, 'res.csv'))

    plot_heatmap(df, os.path.join(args.dir_path, 'res.pdf'))


def cross_domain_retrieval(x_src, y_src, x_tgt, y_tgt, zeroshot=False):
    mAP, prec = evaluate(x_tgt, y_tgt, x_src, y_src)
    txt = ('mAP@all: %.04f Prec@100: %.04f\t' % (mAP, prec))
    print(txt)

    # perform refinement
    g_src_x = KNN(x_src, x_tgt, K=1, mode='ones')

    if zeroshot:
        alpha = 0.7
    else:
        alpha = 0.4
    new_src_x = slerp(alpha, L2norm(x_src), L2norm(g_src_x))
    mAP, prec = evaluate(x_tgt, y_tgt, new_src_x, y_src)
    txt = ('(w. refinement) mAP@all: %.04f Prec@100: %.04f\t' % (mAP, prec))
    print(txt)

    return mAP


def evaluate(im_x, im_y, sk_x, sk_y, return_idx=False):
    idx = retrieve(sk_x, im_x)
    if np.array_equal(sk_x, L2norm(im_x)) or np.array_equal(sk_x, im_x):
        idx = idx[:, 1:]
    prec, mAP = score(sk_y, im_y, idx)
    if return_idx:
        return mAP, prec, idx
    else:
        return mAP, prec


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


def plot_heatmap(df, path,
                 vmin=0, vmax=1, nticks=11,
                 digits="{x:.3f}", cmap="viridis",
                 metric="mAP@all"):
    fig, ax = plt.subplots()

    arr = [0, 1, 2, 5, 4, 3]
    val = df.values[arr, :][:, arr]
    leg = ['clipart', 'infograph', 'painting', 'pencil', 'photo', 'sketch']

    im, cbar = heatmap(val, leg, leg, ax=ax,
                   cmap=cmap, cbarlabel=metric,
                   vmax=0.7,
                   cbar_kw={'boundaries': np.linspace(vmin, vmax, nticks)})
    texts = annotate_heatmap(im, valfmt=digits, textcolors=["white", "black"])

    fig.savefig(path, bbox_inches='tight', pad_inches=0, dpi=300)


if __name__ == '__main__':
    configs = get_config()
    fname = os.path.join(args.dir_path, 'res.csv')
    if os.path.isfile(fname):
        df = pd.read_csv(fname, index_col=0)
        plot_heatmap(df, os.path.join(args.dir_path, 'res.pdf'),
                     vmax=0.7, nticks=8)
    else:
        any2any_retrieval(configs)
