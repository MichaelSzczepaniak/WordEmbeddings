import os
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib

# For get_roc_curves function - these look good in jupyter notebooks,
# but can change these as needed
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14)
plt.rcParams['figure.figsize'] = [10, 10]


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


def get_roc_curves(y_tests, y_scores, model_names=None, plot_title='Comparing Model ROCs', colors=None):
    """
    Creates a plot of ROC curves and their corresponding AUC values for a set of models.

    Args:
          y_tests(np.array(int)): 2-d array where each column are the binary class labels
          (0 or 1) for a particular model. This array MUST have either a single column OR
          the same number of columns as y_scores.
          
          If y_tests is a single column vector, then function assumes that the same y_tests
          values should be a applied to each column in y_scores.
          
          y_scores(np.array(float)): 2-d array where rows are samples and each column are
          the probabilities that each corresponding y_tests value = 1 for a particular model
          
          model_names(list(str)): list of size y_tests.shape[1] = y_scores.shape[1]
          which are the names of the models used to generate each score column. If no model
          names are passed in (default), generic names of the form "model x" will be created
          where x is an integer in [0, y_tests.shape[1])

          colors(list(str)): a list of colors. If None (default) function will use the 10 color
          Tableau pallette: grey or grey, brown, orange, olive, green, cyan, blue, purple, pink, red

    Returns:
        2-tuple: First item is a matplotlib.pyplot object which has a show() method which renders the plot.
        Second item is a dict with keys that are the model_names and values that are the AUC
        of the True Positive Rate vs False Positive Rate (ROC) curve for that model.
                 
    """
    
    # ensure single dim vectors are 1D column vectors so they can be sliced consistenly later on
    if len(y_tests.shape) == 1:
        y_tests = y_tests.reshape(-1, 1)
    if len(y_scores.shape) == 1:
        y_scores = y_scores.reshape(-1, 1)
        
    n_models = y_scores.shape[1]
    
    # check shapes of the true labels (y_test) and model-computed probabilities (y_scores)
    if y_tests.shape[1] > 1 and y_scores.shape[1] != y_tests.shape[1]:
        print("get_roc_curves ERROR: ")
        print("y_tests has {} columns, y_scores has {} columns".format(y_tests.shape[1],
                                                                       y_scores))
        return False
    elif y_tests.shape[1] == 1 and y_scores.shape[1] > 1:
        print("DEBUG get_roc_curves: BEFORE expanding y_tests from 1 to {} columns".format(y_scores.shape[1]))
        # If y_tests is a single column vector and n_models > 1, add copies of the single y_tests column
        y_tests = np.reshape(y_tests, (-1, 1))
        print("DEBUG get_roc_curves: BEFORE expansion, y_tests shape = {}".format(y_tests.shape))
        y_expanded = np.copy(y_tests)
        print("DEBUG get_roc_curves: BEFORE expansion, y_expanded shape = {}".format(y_expanded.shape))
        for i in range(n_models-1):
            y_expanded = np.hstack((y_expanded, y_tests))
            print("DEBUG get_roc_curves: DURING expansion, i = {} ".format(i))
            print("DEBUG get_roc_curves: DURING expansion, y_expanded shape = {} ".format(y_expanded.shape))
        y_tests = y_expanded
        print("DEBUG get_roc_curves: AFTER expansion, y_tests columns = {} ".format(y_tests.shape[1]))
    
    print(f"Comparing {n_models} models")
    # If no model names are passed in, create generic names
    if model_names == None:
        model_names = ['model' + str(i) for i in range(n_models)]
    
    plt.figure()
    lw = 2
    if colors == None:
        # https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
        colors = ['tab:grey', 'tab:blue', 'tab:orange', 'tab:red', 'tab:purple',
                  'tab:green', 'tab:cyan', 'tab:brown', 'tab:olive', 'tab:pink']
    
    color_count = len(colors)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # compute true pos rate and false pos rate over range of thresholds and AUC for each model
    for i in range(n_models):
        fpr[i], tpr[i], _ = roc_curve(y_tests[:, i], y_scores[:, i])
        roc_auc[model_names[i]] = auc(fpr[i], tpr[i])
    
    # plot reference line: random classifier
    plt.plot([0, 1], [0, 1], color=colors[0], lw=lw, linestyle='--')
    # add traces for each model
    for j in range(0, n_models):
        plt.plot(fpr[j], tpr[j], color=colors[j % color_count + 1],
                 lw=lw, label=model_names[j] + ' (AUC = %0.2f)' % roc_auc[model_names[j]])
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(plot_title, fontsize=14)
    plt.legend(loc="lower right", fontsize=14)
    
    return plt, roc_auc


def doc2matrix(doc, voc_embs, emb_dims=300):
    """Converts a string (doc) representing a document into a (emb_dims x n)
    matrix of word embeddings where n is the number of words in doc.
    
    Args:
        doc(str): list(str) where each string is a review (document aka doc) consisting of
        space delimited words
        
        voc_embs(dict): dictionary with keys that are the words in the selected vocabulary and
        values that are the emb_dims-dimensional word embeddings for those words.
        
        emb_dims(int): size of the word embeddings used to encode words in voc_embs
    
    Returns:
        numpy.ndarray: emb_dims x n array of floats representing a doc
        
        n is the number of word tokens in doc
    
    """
    doc_words = doc.split()
    raw_doc = np.zeros((emb_dims, 1))
    # build the d x n matrix of embeddings as a raw representation of doc
    for i, word in enumerate(doc_words):
#         print(word)
        if i == 0:
            if word in voc_embs:
                raw_doc = voc_embs[word].reshape(-1, 1)
            else:
                raw_doc = np.zeros((emb_dims, 1))
                
#             print("initializeing raw_doc: {}".format(raw_doc.shape))
            continue
        else:
            # use zero vector for words that have no embedding
            if word in voc_embs:
                word_emb = voc_embs[word].reshape(-1, 1)
#                 print(word_emb, word_emb.shape)
            else:
                word_emb = np.zeros((emb_dims, 1))
            
#         print(raw_doc.shape, " | ", word_emb.shape)
        raw_doc = np.hstack((raw_doc, word_emb))
        
        
    return raw_doc


def get_x_min_max(doc_as_embs):
    """Creates the coordinate min/max vector representation for a document
    represented as horizontally stacked word embeddings.
    
    Args:
        doc_as_embs (numpy.ndarray): document represented as a d x n matrix
            where d is the size of the word embeddings and n is the number
            of words in the document being represented
    
    Returns:
        numpy.ndarray: real vector column vector of length 2d where d i
        is the size of the word embeddings
    
    """
    
    min_vec = np.amin(doc_as_embs, axis=1).reshape(-1, 1)
    max_vec = np.amax(doc_as_embs, axis=1).reshape(-1, 1)
    x_vec = np.vstack((min_vec, max_vec))
    
    return x_vec


def create_emb_feature_matrix(docs_as_matrices):
    """Converts a list of documents matrices as described in Figure 6.
    into a matrix of document vectors with columns described by
    Figure 7.
    
    Args:
        docs_as_matrices (list): list where each element is a (d x w)
            matrix of floats where d is the length of the embeddings
            vectors representing the words in each document and w is
            the number of word tokens in each document
    
    Returns:
        numpy.ndarray: matrix of floats of shape (2d x n) where n is
        the number of input documents that are to be used as inputs
        to the model
        
    """
    
    # iterate through the list of documents represented as embedding vectors and create the min and max vectors
    for i, doc_as_matrix in enumerate(docs_as_matrices):
        x_vec = get_x_min_max(doc_as_matrix)
#      print(i, x_vec.shape)
        if i == 0:
            x_features = np.copy(x_vec.T)
#             print("intialize x_feature: {}".format(x_features.shape))
        else:
            x_features = np.vstack((x_features, x_vec.T))
#             print("vstack x_feature: {}".format(x_features.shape))
    
    return x_features