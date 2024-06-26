## Fort Collins Data Science Meetup

This project is the code used for the following presentations on **Word Embeddings**:

+ [Part I - August 11, 2020](https://www.meetup.com/Fort-Collins-Data-Science/events/bvzbtrybclbpb)
+ [Part II - September 8, 2020](https://www.meetup.com/Fort-Collins-Data-Science/events/bvzbtrybcmblb)

## Word Embeddings

This project is focused on explaining what word embeddings are and how to use them in the context of text classification.  A jupyter notebook is used as the analysis platform.  Logistic regression and CNN text classifiers (TODO) are built to do sentiment analysis on product reviews using bag-of-words (BOW) and word embeddings (WE's) in order to illustrate these ideas.


## What are word embeddings?

WE's are an excellent example of representational learning (RL) applied to Natural Language Processing (NLP). If the last sentence sounded intimidating, don't worry. We'll unpack what RL is in general and how WE's fit into the picture.

## Goals

My goals for part 1 of this two part series is to help you gain an intuitive understanding of representational learning, provide an overview of RL basics (e.g. encoding and decoding), what WE's are and finally, why we want to build them. The math will be kept to a minimum to keep the focus on building a solid intuition. We'll do this by starting with a familiar example: applying logistic regression to do sentiment classification.

In part 2, armed with some intuition, we'll look at few approaches to how WE's are learned. We'll then revisit our logistic regression problem and see how WE's improves the performance. If there is time, we'll cover some basics of the spaCy library which is an awesome open source set of NLP tools.

## References

### What are Word Embeddings and how are they used?

This is 30 minute video presentation which includes a coding demo: https://www.youtube.com/watch?v=GakXrRqKdzw

### GloVe embeddings

[GloVe: Global Vectors for Word Representation site](https://nlp.stanford.edu/projects/glove/)
  + Pre-trained word vectors where obtained [here](http://nlp.stanford.edu/data/glove.6B.zip)
  
### Additional references listed in the notebook