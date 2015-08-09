import numpy as np
from scipy.sparse import issparse

import logging
logging.basicConfig(level=logging.DEBUG)


class PMICalculator(object):
    """
    Parameter:
    -----------
    doc2word_vectorizer: object that turns list of text into doc2word matrix
        for example, sklearn.feature_extraction.test.CountVectorizer
    """
    def __init__(self, doc2word_vectorizer=None,
                 doc2label_vectorizer=None):
        self._d2w_vect = doc2word_vectorizer
        self._d2l_vect = doc2label_vectorizer

        self.index2word_ = None
        self.index2label_ = None
        
    def from_matrices(self, d2w, d2l, pseudo_count=1):
        """
        Parameter:
        ------------
        d2w: numpy.ndarray or scipy.sparse.csr_matrix
            document-word frequency matrix
        
        d2l: numpy.ndarray or scipy.sparse.csr_matrix
            document-label frequency matrix
            type should be the same with `d2w`

        pseudo_count: float
            smoothing parameter to avoid division by zero

        Return:
        ------------
        numpy.ndarray: #word x #label
            the pmi matrix
        """        
        denom1 = d2w.T.sum(axis=1)
        denom2 = d2l.sum(axis=0)

        # both are dense
        if (not issparse(d2w)) and (not issparse(d2l)):
            numer = np.matrix(d2w.T > 0) * np.matrix(d2l > 0)
            denom1 = denom1[:, None]
            denom2 = denom2[None, :]
        # both are sparse
        elif issparse(d2w) and issparse(d2l):
            numer = ((d2w.T > 0) * (d2l > 0)).todense()
        else:
            raise TypeError('Type inconsistency: {} and {}.\n' +
                            'They should be the same.'.format(
                                type(d2w), type(d2l)))

        # dtype conversion
        numer = np.asarray(numer, dtype=np.float64)
        denom1 = np.asarray(
            denom1.repeat(repeats=d2l.shape[1], axis=1),
            dtype=np.float64)
        denom2 = np.asarray(
            denom2.repeat(repeats=d2w.shape[1], axis=0),
            dtype=np.float64)

        # smoothing
        numer += pseudo_count

        return np.log(d2w.shape[0] * numer / denom1 / denom2)

    def from_texts(self, docs, labels):
        """
        Parameter:
        -----------
        docs: list of list of string
            the tokenized documents

        labels: list of list of string
        
        Return:
        -----------
        numpy.ndarray: #word x #label
            the pmi matrix
        """
        d2w = self._d2w_vect.fit_transform(map(lambda sent: ' '.join(sent),
                                               docs))

        # save it to avoid re-computation
        self.d2w_ = d2w

        d2l = self._d2l_vect.transform(docs, labels)

        # remove the labels without any occurrences
        indices = np.asarray(d2l.sum(axis=0).nonzero()[1]).flatten()
        d2l = d2l[:, indices]

        indices = set(indices)
        labels = [l
                  for i, l in self._d2l_vect.index2label_.items()
                  if i in indices]

        self.index2label_ = {i: l
                             for i, l in enumerate(labels)}

        if len(self.index2label_) == 0:
            logging.warn("After label filtering, there is nothing left.")

        self.index2word_ = {i: w
                            for w, i in self._d2w_vect.vocabulary_.items()}
        return self.from_matrices(d2w, d2l)
