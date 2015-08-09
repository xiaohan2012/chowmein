import numpy as np
from scipy.sparse import csr_matrix
from nose.tools import assert_equal
from numpy.testing import assert_array_almost_equal
from chowmein.pmi import PMICalculator

# 4 words
# 3 documents
d2w = np.asarray([[2, 0, 0, 1],
                  [0, 0, 1, 1],
                  [0, 3, 0, 1]])

d2w_sparse = csr_matrix(d2w)

# 2 labels
# 3 documents
d2l = np.asarray([[1, 0],
                  [0, 2],
                  [1, 1]])
d2l_sparse = csr_matrix(d2l)

cal = PMICalculator()


def test_from_matrices_no_smoothing():
    # some warning will be output
    expected = np.log(3 * np.asarray([[0.25, 0.],
                                      [0.16666667, 0.11111111],
                                      [0., 0.33333333],
                                      [0.16666667, 0.11111111]]))

    # dense input
    assert_array_almost_equal(cal.from_matrices(d2w, d2l, pseudo_count=0),
                              expected)
    # sparse input
    assert_array_almost_equal(cal.from_matrices(d2w_sparse, d2l_sparse,
                                                pseudo_count=0),
                              expected)


def test_from_matrices_with_smoothing():
    expected = np.asarray([[-2.876721e-01, -1.220607e+01],
                           [-6.931372e-01, -1.098602e+00],
                           [-1.110746e+01, 9.999950e-06],
                           [-6.931372e-01, -1.098602e+00]])

    assert_array_almost_equal(cal.from_matrices(d2w, d2l, pseudo_count=1e-5),
                              expected, decimal=5)

from chowmein.tests.test_text import (docs, labels)
from sklearn.feature_extraction.text import CountVectorizer
from chowmein.text import LabelCountVectorizer


def test_from_texts():
    cal = PMICalculator(doc2word_vectorizer=CountVectorizer(min_df=0),
                        doc2label_vectorizer=LabelCountVectorizer())
    actual = cal.from_texts(docs, labels)
    assert_equal(actual.shape[1], 4)
    assert_equal(actual.shape[0], 9)
    assert_equal(cal.index2word_, {0: u'information',
                                   1: u'language',
                                   2: u'learning',
                                   3: u'machine',
                                   4: u'mining',
                                   5: u'natural',
                                   6: u'processing',
                                   7: u'retrieval',
                                   8: u'text'})
    assert_equal(cal.index2label_, {0: 'information retrieval'.split(),
                                    1: 'machine learning'.split(),
                                    2: 'natural language processing'.split(),
                                    3: 'text mining'.split()})


def test_from_texts_nonexisting_label():
    cal = PMICalculator(doc2word_vectorizer=CountVectorizer(min_df=0),
                        doc2label_vectorizer=LabelCountVectorizer())
    actual = cal.from_texts(docs, labels[:2] + [('haha', 'lala')] +
                            labels[2:] + [('non', 'existing')])
    assert_equal(actual.shape[1], 4)
    assert_equal(cal.index2label_, {0: 'information retrieval'.split(),
                                    1: 'machine learning'.split(),
                                    2: 'natural language processing'.split(),
                                    3: 'text mining'.split()})
