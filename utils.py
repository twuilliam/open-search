import cv2
import torch
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import warp, AffineTransform


def L2norm(x):
    return x / x.norm(p=2, dim=1)[:, None]


def cosine_similarity(x, y=None, eps=1e-8):
    if y is None:
        w = x.norm(p=2, dim=1, keepdim=True)
        return torch.mm(x, x.t()) / (w * w.t()).clamp(min=eps)
    else:
        xx = L2norm(x)
        yy = L2norm(y)
        return xx.matmul(yy.t())


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_numpy(x):
    return x.cpu().data.numpy()[0]


def get_backbone(args, pretrained=True):
    from models import VGG16, VGG19, ResNet50, SEResNet50

    if args.backbone == 'resnet':
        output_shape = 2048
        backbone = ResNet50(pretrained=pretrained, kp=args.kp)
    elif args.backbone == 'vgg16':
        output_shape = 4096
        backbone = VGG16(pretrained=pretrained)
    elif args.backbone == 'vgg19':
        output_shape = 4096
        backbone = VGG19(pretrained=pretrained)
    elif args.backbone == 'seresnet':
        output_shape = 2048
        backbone = SEResNet50(pretrained=pretrained)
    return output_shape, backbone


def random_transform(img):
    '''Same augmentation as Qiu et al ICCV 2019
    https://github.com/qliu24/SAKE
    '''
    if img.shape[0] != 224:
        img = cv2.resize(img, (224, 224))

    if np.random.random() < 0.5:
        img = img[:,::-1,:]

    if np.random.random() < 0.5:
        sx = np.random.uniform(0.7, 1.3)
        sy = np.random.uniform(0.7, 1.3)
    else:
        sx = 1.0
        sy = 1.0

    if np.random.random() < 0.5:
        rx = np.random.uniform(-30.0*2.0*np.pi/360.0,+30.0*2.0*np.pi/360.0)
    else:
        rx = 0.0

    if np.random.random() < 0.5:
        tx = np.random.uniform(-10,10)
        ty = np.random.uniform(-10,10)
    else:
        tx = 0.0
        ty = 0.0

    if np.random.random()<0.7:
        aftrans = AffineTransform(scale=(sx, sy), rotation=rx, translation=(tx,ty))
        img_aug = warp(img,aftrans.inverse, preserve_range=True).astype('uint8')
        return img_aug
    else:
        return img


def get_semantic_fname(space='word2vec'):
    if space == 'word2vec':
        return 'word2vec-google-news.npy'
    elif space == 'shrec':
        return 'w2v.npz'


def zero_cnames(dataset):
    if dataset == 'Sketchy':
        # same split at Qiu et al ICCV 2019
        cnames = ['cup', 'chicken', 'camel',
                  'swan', 'squirrel', 'snail', 'scissors',
                  'harp', 'horse',
                  'ray', 'rifle',
                  'pineapple', 'parrot',
                  'volcano',
                  'windmill', 'wine_bottle',
                  'teddy_bear', 'tree', 'tank',
                  'deer',
                  'airplane',
                  'wheelchair',
                  'umbrella',
                  'butterfly', 'bell']
    elif dataset == 'TU-Berlin':
        # same split at Qiu et al ICCV 2019
        cnames = ['banana', 'bottle_opener', 'bus', 'brain', 'bridge', 'bread',
                  'suitcase', 'streetlight', 'shoe', 'snowboard', 'space_shuttle',
                  'tractor', 'telephone', 'teacup', 't_shirt', 'trombone', 'table',
                  'canoe',
                  'fan', 'frying_pan',
                  'penguin', 'pizza', 'parachute',
                  'laptop', 'lighter',
                  'hot_air_balloon', 'horse',
                  'ant',
                  'windmill',
                  'rollerblades']
    elif dataset == 'domainnet':
        # zero-shot split with at least 40 samples per category
        cnames = ['The_Mona_Lisa', 'animal_migration',
                  'bandage', 'beach', 'beard', 'bread',
                  'calendar', 'campfire', 'circle',
                  'door', 'ear', 'eyeglasses',
                  'feather', 'flashlight', 'fork',
                  'garden', 'grass',
                  'hat', 'hockey_stick', 'hot_air_balloon', 'hurricane',
                  'key', 'knee', 'ladder', 'lantern', 'mouth',
                  'octopus', 'onion',
                  'palm_tree', 'picture_frame', 'pond', 'potato',
                  'rake', 'roller_coaster',
                  'sailboat', 'sandwich', 'scissors', 'snowflake', 'steak',
                  'stop_sign', 'string_bean', 'suitcase', 'sun',
                  'tree', 'windmill']
    return cnames


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    (adapted from the matplotlib example)

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.
    (adapted from the matplotlib example)

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
