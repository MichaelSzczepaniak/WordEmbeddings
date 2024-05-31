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


def get_vocab_embeddings(embedding_index, voc):
    """
    Gets the embeddings for the words in a vocabulary
    
    Args:
        embedding_index(dict): a dictionary with keys that are words in the
        embeddings vocabulary and values that are embedding vectors for those words
        
        voc(list(str)): list of words that we want embeddings for
    
    
    Returns(tuple): 2-tuple where the first element is a dict of vocabulary word
    keys and embedding vector values. The second element is a list of the words
    that were not found in the embedding_index dict sorted in alphabetical order
    
    """

    # get embeddings for just the words in our vocabulary
    voc_embeddings = {}
    no_embeddings = []
    for word in voc:
        emb = embedding_index.get(word, None)
        if type(emb) == np.ndarray:
            voc_embeddings[word] = embedding_index.get(word, None)
        else:
            no_embeddings.append(word)
#         print("No embedding for {}".format(word))
    
    return voc_embeddings, sorted(no_embeddings)



def word_NN(w, vocab_embeddings):
    """
    Finds the word closest to w in the vocabulary that isn't w itself.
    
    Args:
        w(str): string, word to compute nearest neighbor for - must be in a key in vocab_embeddings
        vocab_embeddings(dict): dictionary with keys that are words in the vocabulary
          and values that are d-dimensional numpy array of floats that are the real-
          vector embeddings for each word in the vocabulary
          
    Returns:
        string: the word in the vocabulary that is the closest to this particular word
    
    """
    
    vocab_words = set(vocab_embeddings.keys())
    # check if the word passed in is in the vocabulary
    if not(w in vocab_words):
        print ("Unknown word")
        return
    
    # remove the word we are looking for the nearest neighbor of
    vocab_words.discard(w)
    vocab_words = list(vocab_words)
    
    # get the embedding for passed in word
    w_embedding = vocab_embeddings[w]
    neighbor = 0
    curr_dist = np.linalg.norm(w_embedding - vocab_embeddings[vocab_words[0]])
    # iterate through all the words in the vocabulary and find the 'closest'
    for i in vocab_words:
        dist = np.linalg.norm(w_embedding - vocab_embeddings[i])
        if (dist < curr_dist):
            neighbor = i
            curr_dist = dist
            
    return neighbor


def embedding_NN(w_embedding, vocab_embeddings, discard_words=[]):
    """
    Finds the word closest to w in the vocabulary that isn't w itself.
    
    Args:
        w_embedding(numpy.ndarray): embedding vector to compute nearest neighbor
        vocab_embeddings(dict): dictionary with keys that are words in the vocabulary
          and values that are d-dimensional numpy array of floats that are the real-
          vector embeddings for each word in the vocabulary
        discard_words(list(str)): list of words to discard from calculation
          
    Returns:
        string: the word in the vocabulary that is the closest to this particular word
    
    """
    
    vocab_words = set(vocab_embeddings.keys())
    if len(discard_words) > 0:
        for discard_word in discard_words:
            vocab_words.discard(discard_word)
    
    # convert vocabulary words to a list
    vocab_words = list(vocab_words)
    
    # initialize neighbor guess and distance from passed in embedding and first
    # word in the vocabulary
    neighbor = ""
    curr_dist = np.linalg.norm(w_embedding - vocab_embeddings[vocab_words[0]])
    # iterate through all the words in the vocabulary and find the 'closest'
    for vword in vocab_words:
        dist = np.linalg.norm(w_embedding - vocab_embeddings[vword])
        if (dist < curr_dist):
            neighbor = vword
            curr_dist = dist
            
    return neighbor