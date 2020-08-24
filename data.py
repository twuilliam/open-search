import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils import zero_cnames


def image_loader(path):
    return cv2.imread(path)[:, :, ::-1]


def sketch_loader(path):
    return cv2.imread(path)[:, :, ::-1]


def create_splits(path, overwrite=False, gzsl=False):
    '''Create Train and Test splits'''
    splits = {}
    for modality in ['im', 'sk']:
        splits[modality] = {}

        fname = os.path.join(path, modality + '.hdf5')
        df = pd.read_hdf(fname)

        # get zero-shot class names
        if overwrite:
            if 'Sketchy' in path:
                dataset = 'Sketchy'
            elif 'TU-Berlin' in path:
                dataset = 'TU-Berlin'
            cnames = zero_cnames(dataset)
            cond = df['cat'].isin(cnames)

            df.loc[~cond, 'split'] = 'train'
            df.loc[cond, 'split'] = 'test'

            if gzsl:
                np.random.seed(1234)
                fnames = df.loc[df['split'] == 'train'].index
                to_select = np.random.choice(fnames,
                                             size=int(len(fnames)*0.2),
                                             replace=False)
                cond = df.index.isin(to_select)
                df.loc[cond, 'split'] = 'test'

            df_train = df.loc[df['split'] == 'train']
            df_train = df_train.assign(cat=df_train['cat'].astype('category'))

            df_test = df.loc[df['split'] == 'test']
            df_test = df_test.assign(cat=df_test['cat'].astype('category'))

            df_gal = df.loc[df['split'] == 'test']
            df_gal = df_gal.assign(cat=df_gal['cat'].astype('category'))

        else:
            df_train = df.loc[df['split'] == 'train']
            df_train = df_train.assign(cat=df_train['cat'].astype('category'))

            df_val = df.loc[df['split'] == 'val']
            df_val = df_val.assign(cat=df_val['cat'].astype('category'))

            df_test = df.loc[df['split'] == 'test']
            df_test = df_test.assign(cat=df_test['cat'].astype('category'))

            df_gal = pd.concat([df_val, df_test])
            df_gal = df_gal.assign(cat=df_gal['cat'].astype('category'))

        splits[modality]['train'] = df_train
        splits[modality]['test'] = df_test
        splits[modality]['gal'] = df_gal
    return splits


def is_ext(fnames):
    return [True if 'ext' in os.path.basename(f) else False for f in fnames]


def create_fewshot_splits(path, subsample=True):
    '''Create Train and Test splits
    Following Hu et al, CVPR 2018
    '''
    test_classes = ['car_(sedan)', 'pear', 'deer', 'couch', 'duck',
                    'airplane', 'cat', 'mouse', 'seagull', 'knife']

    splits = {}
    for modality in ['im', 'sk']:
        splits[modality] = {}

        fname = os.path.join(path, modality + '.hdf5')
        df = pd.read_hdf(fname)

        if subsample:
            # subsampling extended images to match Hu et al CVPR18
            np.random.seed(1234)

            # get how many to discard
            cond = is_ext(df.index)
            df['ext'] = cond
            vv, cc = np.unique(df.loc[cond, 'cat'], return_counts=True)
            n_select = np.asarray(np.round(cc / float(np.sum(cc)) * 4336), dtype=int)

            # collect fnames to discard
            to_remove = []
            for v, n in zip(vv, n_select):
                idx = df[(df['ext'] == True) & (df['cat'] == v)].index
                to_remove.extend(np.random.choice(idx, size=n, replace=False))

            # subsampled df
            df = df[~df.index.isin(to_remove)]

        cond = df['cat'].isin(test_classes)

        df_train = df.loc[~cond]
        df_train = df_train.assign(cat=df_train['cat'].astype('category'))

        df_test = df.loc[cond]
        df_test = df_test.assign(cat=df_test['cat'].astype('category'))

        splits[modality]['train'] = df_train
        splits[modality]['test'] = df_test
    return splits


def create_shape_splits(path):
    '''Create Train and Test splits for 3D shapes'''
    splits = {}
    for modality in ['cad', 'sk']:
        splits[modality] = {}

        fname = os.path.join(path, modality + '.hdf5')
        df = pd.read_hdf(fname)

        if 'split' in df.columns:
            df_train = df.loc[df['split'] == 'train']
            df_train = df_train.assign(cat=df_train['cat'].astype('category'))

            df_test = df.loc[df['split'] == 'test']
            df_test = df_test.assign(cat=df_test['cat'].astype('category'))
        else:
            df_train = df.copy()
            df_train = df_train.assign(cat=df_train['cat'].astype('category'))

            df_test = df.copy()
            df_test = df_test.assign(cat=df_test['cat'].astype('category'))

        splits[modality]['train'] = df_train
        splits[modality]['test'] = df_test
        splits[modality]['gal'] = df_test

        if modality == 'cad':
            splits['im'] = {}
            splits['im']['train'] = df_train
            splits['im']['gal'] = df_test
            splits['im']['test'] = df_test
    return splits


def create_multi_splits(path, domain, overwrite=False):
    '''Create Train and Test splits for DomainNet'''
    splits = {}
    for modality in ['im', 'sk']:
        splits[modality] = {}

        fname = os.path.join(path, modality + '.hdf5')
        df = pd.read_hdf(fname)

        if modality == 'im':
            cond = df['domain'] == domain
            df = df.loc[cond]

        if overwrite:
            dataset = 'domainnet'
            cnames = zero_cnames(dataset)
            cond = df['cat'].isin(cnames)

            df.loc[~cond, 'split'] = 'train'
            df.loc[cond, 'split'] = 'test'

        cond = df['split'] == 'train'

        df_train = df.loc[cond]
        df_train = df_train.assign(cat=df_train['cat'].astype('category'))

        df_test = df.loc[~cond]
        df_test = df_test.assign(cat=df_test['cat'].astype('category'))

        splits[modality]['train'] = df_train
        splits[modality]['test'] = df_test
        splits[modality]['gal'] = df_test
    return splits


class DataLoader(Dataset):
    def __init__(self, split, transform, root='', mode='im'):
        self.split = split
        self.transform = transform
        self.root = root
        if mode == 'im':
            self.loader = image_loader
        elif mode == 'sk':
            self.loader = sketch_loader

    def __getitem__(self, index):
        """ Read img, transform img and return class label in long int """
        # read img and apply transformations
        fname = self.split.iloc[index].name
        img = self.loader(os.path.join(self.root, fname))
        img = self.transform(img)

        # get class label
        item = self.split['cat'].cat.codes.iloc[index].astype('int64')
        return img, item

    def __len__(self):
        return self.split.shape[0]


def get_proxies(path_semantic, class_names):
    try:
        semantic = np.load(path_semantic, allow_pickle=True).item()
    except:
        if os.path.splitext(path_semantic)[-1] == '.npz':
            semantic = np.load(path_semantic)['wv'].item()
        elif os.path.splitext(path_semantic)[-1] == '.pkl':
            import pickle
            with open(path_semantic, 'rb') as f:
                semantic = pickle.load(f)
        else:
            semantic = np.load(path_semantic).reshape(-1)[0]
    proxies = np.stack([semantic[c] for c in class_names])
    return np.float32(proxies)
