import numpy as np
from scipy.sparse import csr_matrix
from nose.tools import assert_equal
from numpy.testing import assert_array_almost_equal
from tmlabel.pmi import PMICalculator

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


def test_from_matrices():
    expected = np.log(3 * np.asarray([[0.25, 0.],
                                      [0.16666667, 0.11111111],
                                      [0., 0.33333333],
                                      [0.16666667, 0.11111111]]))
    # dense input
    assert_array_almost_equal(cal.from_matrices(d2w, d2l), expected)
    # sparse input
    assert_array_almost_equal(cal.from_matrices(d2w_sparse, d2l_sparse),
                              expected)


from test_text import (docs, labels)
from sklearn.feature_extraction.text import CountVectorizer
from tmlabel.text import LabelCountVectorizer


def test_from_texts():
    cal = PMICalculator(doc2word_vectorizer=CountVectorizer(min_df=0),
                        doc2label_vectorizer=LabelCountVectorizer())
    actual = cal.from_texts(docs, labels)
    assert_equal(actual.shape[1], 4)
    assert_equal(actual.shape[0], 10)
    assert_equal(cal.index2word_, {0: u'information',
                                   1: u'language',
                                   2: u'learning',
                                   3: u'learningmachine',
                                   4: u'machine',
                                   5: u'mining',
                                   6: u'natural',
                                   7: u'processing',
                                   8: u'retrieval',
                                   9: u'text'})
    assert_equal(cal.index2label_, {0: 'information retrieval',
                                    1: 'machine learning',
                                    2: 'natural language processing',
                                    3: 'text mining'})
