import torch
import faiss
import numpy as np
import multiprocessing as mp
from torch.autograd import Variable
from metrics import precision_at_k, mean_average_precision, average_precision
from metrics import dcg_at_k, ndcg_at_k
from utils import AverageMeter, to_numpy


def L2norm(x):
    return x / np.linalg.norm(x, axis=1)[:, None]


def extract(loader, model):
    model.eval()
    outputs = []
    labels = []
    for i, (imgs, l) in enumerate(loader):
        if torch.cuda.is_available():
            imgs = imgs.cuda(async=True)

        imgs = Variable(imgs, volatile=True)
        outputs.append(model(imgs).data.cpu().numpy())

        labels.extend(l)
    return np.vstack(outputs), np.asarray(labels)


def extract_predict(loader, model, proxies, criterion):
    model.eval()
    outputs = []
    labels = []

    val_acc = AverageMeter()

    copy = False

    for i, (imgs, l) in enumerate(loader):
        if len(imgs.shape) == 5:
            batch_size, nviews = imgs.shape[0], imgs.shape[1]
            imgs = imgs.view(batch_size * nviews, 3, 224, 224)

            # hack to handle multiple gpus
            # can crash if there are 0 im
            if batch_size < 4:
                imgs = torch.cat([imgs, imgs], dim=0)
                l = torch.cat([l, l])
                copy = True

        labels.extend(l)

        if torch.cuda.is_available():
            imgs = imgs.cuda(async=True)
            l = l.cuda(async=True)

        imgs = Variable(imgs, volatile=True)
        if torch.cuda.device_count() > 1:
            embs = model(imgs)
        else:
            embs = model(imgs)

        if copy:
            embs = embs[:batch_size]
            l = l[:batch_size]

        outputs.append(embs.data.cpu().numpy())

        loss, acc = criterion(embs, l, proxies)
        val_acc.update(acc, imgs.size(0))

    return np.vstack(outputs), np.asarray(labels), val_acc.avg


def retrieve(query, gallery, dist='euc', L2=True):
    d = query.shape[1]
    if dist == 'euc':
        index_flat = faiss.IndexFlatL2(d)
    elif dist == 'cos':
        index_flat = faiss.IndexFlatIP(d)

    if L2:
        query = L2norm(query)
        gallery = L2norm(gallery)

    index_flat.add(gallery)
    K = gallery.shape[0]
    D, I = index_flat.search(query, K)
    return I


def KNN(query, gallery, K=10, mode='ones'):
    '''retrieves the K-Nearest Neighbors in the gallery'''
    d = query.shape[1]
    query = L2norm(query)
    gallery = L2norm(gallery)

    res = faiss.StandardGpuResources()

    index_flat = faiss.IndexFlatL2(d)

    if torch.cuda.is_available():
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        gpu_index_flat.add(gallery)
        D, I = gpu_index_flat.search(query, K)
    else:
        index_flat.add(gallery)
        D, I = index_flat.search(query, K)

    if mode == 'lin':
        weights = (float(K) - np.arange(0, K)) / float(K)
    elif mode == 'exp':
        weights = np.exp(-np.arange(0, K))
    elif mode == 'ones':
        weights = np.ones(K)
    weights_sum = weights.sum()

    new_queries = []
    for i in range(len(query)):
        idx = I[i, :K]
        to_consider = gallery[idx, :]
        new_queries.append(np.dot(weights, to_consider) / weights_sum)
    new_queries = np.asarray(new_queries, dtype=np.float32)
    return new_queries


def score(sk_labels, im_labels, index):
    res = np.equal(im_labels[index], sk_labels[:, None])

    prec = np.mean([precision_at_k(r, 100) for r in res])

    pool = mp.Pool(processes=10)
    results = [pool.apply_async(average_precision, args=(r,)) for r in res]
    mAP = np.mean([p.get() for p in results])
    pool.close()
    return prec, mAP


def score_shape(sk_labels, im_labels, index):
    vv, cc = np.unique(im_labels, return_counts=True)
    lut = {}
    for v, c in zip(vv, cc):
        lut[v] = c

    res = np.equal(im_labels[index], sk_labels[:, None])

    # 1-NN
    nn = np.mean(res[:, 0])

    # first and second tier
    ft = np.mean([np.sum(r[:lut[l]]) / float(lut[l])
                  for r, l in zip(res, sk_labels)])
    st = np.mean([np.sum(r[:2 * lut[l]]) / float(lut[l])
                  for r, l in zip(res, sk_labels)])

    # e-measure
    prec = np.mean([precision_at_k(r, 32) for r in res])
    rec = np.mean([np.sum(r[:32]) / float(lut[l])
                   for r, l in zip(res, sk_labels)])
    e_measure = 2 * prec * rec / (prec + rec)

    # dcg
    pool = mp.Pool(processes=10)
    results = [pool.apply_async(dcg_at_k, args=(r, len(im_labels), 1)) for r in res]
    mDCG = np.mean([p.get() for p in results])
    pool.close()

    # ndgc
    pool = mp.Pool(processes=10)
    results = [pool.apply_async(ndcg_at_k, args=(r, len(im_labels), 1)) for r in res]
    mnDCG = np.mean([p.get() for p in results])
    pool.close()

    # map
    pool = mp.Pool(processes=10)
    results = [pool.apply_async(average_precision, args=(r,)) for r in res]
    mAP = np.mean([p.get() for p in results])
    pool.close()

    return nn, ft, st, e_measure, mnDCG, mAP


def score_single(sk_labels, im_labels, index):
    res = np.equal(im_labels[index], sk_labels[:, None])

    prec = np.mean([precision_at_k(r, 100) for r in res])
    mAP = mean_average_precision(res)
    return prec, mAP
