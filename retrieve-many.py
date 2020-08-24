import argparse
import os
import json
import faiss
import numpy as np
import pandas as pd
from data import create_multi_splits
from validate import L2norm
from validate import retrieve, KNN, score


# Training settings
parser = argparse.ArgumentParser(description='PyTorch SBIR')
parser.add_argument('--dir-path', type=str, default='exp', metavar='ED',
                    help='directory with domainnet models')
parser.add_argument('--new-data-path', type=str, default='', metavar='ED',
                    help='overwrite data path')
parser.add_argument('--eval', type=str, required=True, metavar='ED',
                    help='many2any|any2many')
args = parser.parse_args()

GROUPS = 500
SEED = 1234


def get_config():
    # iterate over folders in the directory
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

                if not args.new_data_path == '':
                    configs[tmp['domain']]['data_dir'] = args.new_data_path
            else:
                configs['quickdraw'] = tmp
                configs['quickdraw']['working_path'] = os.path.join(
                    args.dir_path, path)
                if not args.new_data_path == '':
                    configs['quickdraw']['data_dir'] = args.new_data_path

    return configs


def get_splits(configs):
    keys = configs.keys()
    keys.sort()

    fpaths = []
    domains = []
    y = []
    for key in keys:
        # get data splits
        df_dir = os.path.join('aux', 'data', configs[key]['dataset'])
        splits = create_multi_splits(df_dir, configs[key]['domain'])
        if key == 'quickdraw':
            fpaths.extend(splits['sk']['test'].index.values)
            domains.extend(splits['sk']['test']['domain'].values)
            y.extend(splits['sk']['test']['cat'].values)
        else:
            fpaths.extend(splits['im']['test'].index.values)
            domains.extend(splits['im']['test']['domain'].values)
            y.extend(splits['im']['test']['cat'].values)
    df = pd.DataFrame({'domain': domains, 'cat': y}, index=fpaths)
    return df


def read_data(fpath):
    data = np.load(fpath)
    return data['features'], data['labels']


def mix_queries(base, complement, alpha=0.5):
    idx = sample_complement(base['y'], complement['y'])
    mixture = alpha * base['x'] + (1-alpha) * complement['x'][idx, :]
    return mixture, idx


def sample_complement(y_base, y_complement):
    np.random.seed(SEED)
    idx = []
    for y in y_base:
        cond_idx = np.argwhere(y_complement == y).squeeze()
        idx.append(np.random.choice(cond_idx))
    return idx


def many2any_retrieval(configs, sources=['quickdraw', 'real']):
    keys = configs.keys()
    keys.sort()

    source_data = {}

    for domain in sources:
        dirname = configs[domain]['working_path']
        fpath = os.path.join(dirname, 'features.npz')

        x, y = read_data(fpath)

        source_data[domain] = {}
        source_data[domain]['x'] = x
        source_data[domain]['y'] = y

    # save images that have been mixed, such that they don't get retrived
    x_src, idx = mix_queries(source_data[sources[0]], source_data[sources[1]])
    y_src = source_data[sources[0]]['y']
    np.save('plop.npy', idx)

    res = {}
    for domain in keys:
        dirname = configs[domain]['working_path']
        fpath = os.path.join(dirname, 'features.npz')

        x_tgt, y_tgt = read_data(fpath)

        if sources[0] == domain and sources[1] == domain:
            pass
        else:
            print('\nRetrieval from %s+%s to %s' %
                  (sources[0], sources[1], domain))

            if domain == sources[1]:
                do_mixture = True
            else:
                do_mixture = False

            tmp = cross_domain_retrieval(
                x_src, y_src, x_tgt, y_tgt,
                zeroshot=configs[domain]['overwrite'],
                mixture=do_mixture)
            res[domain] = tmp

    os.remove('plop.npy')


def get_data(configs):
    keys = configs.keys()
    keys.sort()

    feats = []
    labels = []
    domains = []
    for i, key in enumerate(keys):
        dirname = configs[key]['working_path']
        fpath = os.path.join(dirname, 'features.npz')

        data = np.load(fpath)
        nsamples = len(data['labels'])

        feats.extend(data['features'])
        labels.extend(data['labels'])
        domains.extend([key] * nsamples)

    return feats, labels, domains


def one2many_retrieve_intent_aware(feats, labels, domains, splits,
                                   source='quickdraw',
                                   zeroshot=False):
    cond = np.asarray(domains) == source

    x_src = np.asarray(feats)[cond, :]
    y_src = np.asarray(labels)[cond]
    x_tgt = np.asarray(feats)[~cond, :]
    y_tgt = np.asarray(labels)[~cond]

    d_tgt = np.asarray(domains)[~cond]

    # KNN
    g_src_x = KNN(x_src, x_tgt, K=1, mode='ones')

    if zeroshot:
        alpha = 0.7
    else:
        alpha = 0.4
    x_src = slerp(alpha, L2norm(x_src), L2norm(g_src_x))

    idx = myretrieve(x_src, x_tgt, topK=100)

    yd_tgt = np.char.add(y_tgt.astype(d_tgt.dtype), d_tgt)

    domains = np.unique(d_tgt)
    categories = np.unique(y_tgt)

    # compute occurrences of every category per domain
    occ = []
    for d in domains:
        occ_inner = []
        for c in categories:
            cond = np.logical_and(d_tgt == d, y_tgt == c)
            occ_inner.append(np.sum(cond))
        occ.append(occ_inner)
    occ = np.asarray(occ, dtype=np.float)

    # normalize occurences
    occ /= np.sum(occ, axis=0)

    import multiprocessing as mp
    from metrics import average_precision

    # compute intent-aware mAP per domain
    mAP_ia = []
    for d in domains:
        yd_src = np.char.add(y_src.astype(d_tgt.dtype), d)
        res = np.char.equal(yd_tgt[idx], yd_src[:, None])
        pool = mp.Pool(processes=10)
        results = [pool.apply_async(average_precision, args=(r,)) for r in res]
        mAP = np.asarray([p.get() for p in results])
        pool.close()

        mAP_ia.append(mAP)

        print('%s: %.3f' % (d, np.mean(mAP)))
    mAP_ia = np.asarray(mAP_ia)

    mAP_ia_final = (occ[:, y_src] * mAP_ia).sum(0).mean()
    print('mAP-IA: %.3f' % mAP_ia_final)

    return idx


def cross_domain_retrieval(x_src, y_src, x_tgt, y_tgt,
                           zeroshot=False, mixture=False):
    mAP, prec = evaluate(x_tgt, y_tgt, x_src, y_src, mixture=mixture)

    txt = ('mAP@all: %.04f Prec@100: %.04f\t' % (mAP, prec))
    print(txt)

    g_src_x = KNN(x_src, x_tgt, K=1, mode='ones')

    if zeroshot:
        alpha = 0.7
    else:
        alpha = 0.4
    new_src_x = slerp(alpha, L2norm(x_src), L2norm(g_src_x))
    mAP, prec = evaluate(x_tgt, y_tgt, new_src_x, y_src, mixture=mixture)
    txt = ('mAP@all: %.04f Prec@100: %.04f\t' % (mAP, prec))
    tmp = '(w. refinement)' % alpha
    txt = tmp + ' ' + txt
    print(txt)

    return mAP


def evaluate(im_x, im_y, sk_x, sk_y, K=False, return_idx=False, mixture=False):
    if not K:
        idx = retrieve(sk_x, im_x)
    else:
        idx = myretrieve(sk_x, im_x, topK=K)

    if mixture:
        selection = np.load('plop.npy')
        rows, cols = idx.shape
        idx = idx[idx != selection[:, None]].reshape(rows, -1)

    prec, mAP = score(sk_y, im_y, idx)
    if return_idx:
        return mAP, prec, idx
    else:
        return mAP, prec


def myretrieve(query, gallery, dist='euc', L2=True, topK=101):
    d = query.shape[1]
    if dist == 'euc':
        index_flat = faiss.IndexFlatL2(d)
    elif dist == 'cos':
        index_flat = faiss.IndexFlatIP(d)

    if L2:
        query = L2norm(query)
        gallery = L2norm(gallery)

    index_flat.add(gallery)
    D, I = index_flat.search(query, topK)
    return I


def get_stats(splits):
    domains = splits['domain'].unique()
    categories = splits['cat'].unique()

    stats = {}
    for c in categories:
        stats[c] = {}
        total = 0.
        for d in domains:
            cond = np.logical_and(splits['domain'] == d, splits['cat'] == c)
            stats[c][d] = np.sum(cond)
            total += np.sum(cond)
        for d in domains:
            stats[c][d] /= total
    return stats


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


if __name__ == '__main__':
    configs = get_config()
    if args.eval == 'many2any':
        many2any_retrieval(configs, sources=['quickdraw', 'quickdraw'])
        many2any_retrieval(configs, sources=['quickdraw', 'infograph'])
        many2any_retrieval(configs)

        many2any_retrieval(configs, sources=['clipart', 'clipart'])
        many2any_retrieval(configs, sources=['clipart', 'quickdraw'])
        many2any_retrieval(configs, sources=['clipart', 'infograph'])

        many2any_retrieval(configs, sources=['real', 'real'])
        many2any_retrieval(configs, sources=['real', 'quickdraw'])
        many2any_retrieval(configs, sources=['real', 'infograph'])

    elif args.eval == 'any2many':
        feats, labels, domains = get_data(configs)
        splits = get_splits(configs)

        one2many_retrieve_intent_aware(feats, labels, domains, splits)
