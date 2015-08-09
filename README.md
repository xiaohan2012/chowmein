[![Build Status](https://travis-ci.org/xiaohan2012/chowmein.svg?branch=master)](https://travis-ci.org/xiaohan2012/chowmein)
[![Coverage Status](https://coveralls.io/repos/xiaohan2012/chowmein/badge.svg?branch=master&service=github)](https://coveralls.io/github/xiaohan2012/chowmein?branch=master)

# chowmein

Automatic labeling of topic models.

The alogirithm is described in [Automatic Labeling of Multinomial Topic Models](http://sifaka.cs.uiuc.edu/czhai/pub/kdd07-label.pdf)

# Example 

We model the abstracts of `NIPS 2014`(NIPS abstracts from 2008 to 2014 is available under `datasets/`).
Meanwhile, we contrain the labels to be tagged as `NN,NN` or `JJ,NN` and use the top 200 most informative labels.


```
>>> python label_topic.py --line_corpus_path datasets/nips-2014.dat --preprocessing wordlen tag --label_tags NN,NN JJ,NN --n_cand_labels 200
...
Topical words:
--------------------
Topic 0: model data framework clustering information distributions two number world propose noise real work small
Topic 1: learning algorithm time problem online regret information decision conditional new stochastic algorithms selection problems
Topic 2: algorithm algorithms problem results learning optimal show function class functions graph bounds based general
Topic 3: learning training networks data tasks features neural kernel performance classification model datasets feature deep
Topic 4: matrix method sparse convex problems methods dimensional problem rank analysis propose regression norm gradient
Topic 5: model models inference approach data linear based gaussian method methods process sampling structure time

Topical labels:
--------------------
Topic labels:
Topic 0: neural population, inference algorithm, likelihood estimator, stochastic optimization, matrix recovery, paper develop, empirical study, covariance matrix
Topic 1: bandit problem, near-optimal regret, function approximation, paper consider, general class, multi-armed bandit, value function, statistical learning
Topic 2: logarithmic factor, statistical learning, convergence rate, communication cost, other hand, main result, solution quality, function approximation
Topic 3: pascal voc, major challenge, natural language, paper introduce, object recognition, policy search, empirical study, classification accuracy
Topic 4: low-rank tensor, low-rank matrix, matrix recovery, coordinate descent, problem finding, direction method, statistical learning, risk minimization
Topic 5: inference algorithm, introduce novel, exponential family, probabilistic inference, neural population, value function, policy search, other hand
```

# Usage


## Command line

For example:

    >>> python label_topic.py --line_corpus_path datasets/nips-2014.dat  --preprocess wordlen tag --label_tags NN,NN

For more details:

    >>> python label_topic.py --help

## Programmatically

Please refer to `label_topic.py`.


# How it works

The current version goes through the following steps

1. Preprocessing using [nltk](http://www.nltk.org/)'s `word_tokenize`, `stem` and `pos_tag`.
1. Candidate phrase detection using *pointwise mutual information*: POS tag constraint can be applied. For now, only **bigrams** are considered.
2. Topic modeling using [LDA](https://pypi.python.org/pypi/lda).
3. Candidate label ranking using the algorithm [here](http://sifaka.cs.uiuc.edu/czhai/pub/kdd07-label.pdf).


# TODO


- Better phrase detection thorugh better POS tagging
- Better ways to compute language models for labels to support `intra topical coverage` heuristic(which is now **disabled**)
- Support for user defined candidate labels
- Faster PMI computation(using Cythong for example)
- More flexibity/option on preprocessing
- Leveraging knowledge base to refine the labels
