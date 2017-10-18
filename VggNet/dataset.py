import itertools as it
import numpy as np
import tensorflow as tf
import pickle 

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def get_cifar10(batch_size=16):
    print("loading yunsoo data ... ")
    
    with open("cifared_dataset/train.pickle", 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        
    trn_pixels = data['data']
    trn_labels = np.array(data['labels'])
    trn_pixels, trn_labels = unison_shuffled_copies(trn_pixels, trn_labels)
    trn_pixels = trn_pixels.reshape(-1, 3, 32, 32).astype(np.float32)

    with open("cifared_dataset/val.pickle", 'rb') as f:
        tst_data = pickle.load(f, encoding='latin1')
        
    tst_pixels = tst_data['data']
    tst_labels = np.array(tst_data['labels'])
    tst_pixels, tst_labels = unison_shuffled_copies(tst_pixels, tst_labels)
    tst_pixels = tst_pixels.reshape(-1, 3, 32, 32).astype(np.float32)

    print("-- trn shape = %s" % list(trn_pixels.shape))
    print("-- tst shape = %s" % list(tst_pixels.shape))

    # transpose to tensorflow's bhwc order assuming bchw order
    trn_pixels = trn_pixels.transpose(0, 2, 3, 1)
    tst_pixels = tst_pixels.transpose(0, 2, 3, 1)

    trn_set = batch_iterator(it.cycle(zip(trn_pixels, trn_labels)), batch_size, cycle=True, batch_fn=lambda x: zip(*x))
    tst_set = (tst_pixels, np.array(tst_labels))

    return trn_set, tst_set

def batch_iterator(iterable, size, cycle=False, batch_fn=lambda x: x):
    """
    Iterate over a list or iterator in batches
    """
    batch = []

    # loop to begining upon reaching end of iterable, if cycle flag is set
    if cycle is True:
        iterable = it.cycle(iterable)

    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch_fn(batch)
            batch = []

    if len(batch) > 0:
        yield batch_fn(batch)
