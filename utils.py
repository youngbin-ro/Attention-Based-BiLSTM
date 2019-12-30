import numpy as np
import argparse
import os
import json
from gensim.models import Word2Vec


def get_preweight(w2v_path, tokenizer, rand_init=False):
    w2v_model = Word2Vec.load(w2v_path)
    embedding_size = w2v_model.wv.vectors.shape[1]
    embedding = np.zeros((len(tokenizer.word2idx), embedding_size))
    for w, i in tokenizer.word2idx.items():
        try:
            embedding[i] += w2v_model.wv.vectors[w2v_model.wv.index2word.index(w)]
        except ValueError:
            if rand_init:
                embedding[i] += np.random.normal(scale=0.6, size=(embedding_size,))
    return embedding


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def save_config(config, print_config=True):
    if print_config:
        for key, value in config.__dict__.items():
            print("{}: {}".format(key, value))
        print()

    with open(os.path.join(config.result_save_path, 'config.txt'), 'w') as fp:
        json.dump(config.__dict__, fp, indent=2)
