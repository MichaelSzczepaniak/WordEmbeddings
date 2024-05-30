import os
import numpy as np


def get_glove(glove_dir, emb_dim, enc = "utf8"):
    """
    Reads in and returns a specified set of GloVe embeddings.
    
    Args:
        glove_dir(str): directory containing the GloVe embeddings
        
        emb_dim(int): dimensions of the embeddings to read in, valid values: 50, 100, 200, 300
        
        enc(str): encoding used to read the embeddings file, default = "utf8"
    
    
    Returns(dict): a dictionary with keys that are words in the embeddings vocabulary
    and values that are emb_dim-dimensional embedding vectors for those words
    
    """

    print('Indexing word vectors.')
    # load the embeddings into a dict with keys that are words and
    # values are the embedding vectors for those words
    embedding_index = {}

    with open(os.path.join(glove_dir, 'glove.6B.' + str(emb_dim) + 'd.txt'), encoding=enc) as f:
        for line in f:
            word, coeffs = line.split(maxsplit = 1)
            coeffs = np.fromstring(coeffs, dtype='float', sep=' ')
            embedding_index[word] = coeffs
        
    print("Found {} word vectors.".format(len(embedding_index)))
    
    return embedding_index