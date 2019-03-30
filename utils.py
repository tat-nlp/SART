import io
import numpy as np


# Read embeddings to a numpy array and word2id dictionary
def read_embeddings(emb_path):
    words2ids = {}
    vectors = None
    # load pre-trained embeddings
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                split = line.split()
                vectors = np.empty([int(split[0]), int(split[1])])
            else:
                word, vect = line.rstrip().split(' ', 1)
                vect = np.fromstring(vect, sep=' ')
                vectors[len(words2ids)] = vect
                words2ids[word] = len(words2ids)
    return vectors, words2ids


# Normalize embeddings
def normalize_embeddings(emb):
    norm = np.linalg.norm(emb, axis=1)
    return emb / norm[:, None]


# Read lines from .txt file into a list of lists of words
def read_lines(filepath):
    split_lines = []
    for line in open(filepath, 'r', encoding="utf8"):
        split_lines.append(line.split())
    return split_lines
