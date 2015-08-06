import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_equal
from tmlabel.text import LabelCountVectorizer

ir = 'information retrieval'
ml = 'machine learning'
nlp = 'natural language processing'
tm = 'text mining'

other = ' --- '

docs = [' '.join([ml, other, ml, nlp]),
        ','.join([ir, tm, other, tm]),
        ' '.join([ml, other, tm, other]),
        ','.join([nlp, nlp, nlp, ir, other]),
        '',
        ' '.join([other, other, other]),
        ''.join([ml, ml, ml])]

labels = [ml, nlp, tm, ir]


def test_label_count_vectorizer():
    vect = LabelCountVectorizer()
    d2l_mat = vect.transform(docs, labels)
    assert_array_equal(d2l_mat.todense(),
                       np.asarray([[0, 2, 1, 0],
                                   [1, 0, 0, 2],
                                   [0, 1, 0, 1],
                                   [1, 0, 3, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 3, 0, 0]], dtype=np.int64))

    assert_equal(vect.index2label_, {0: ir, 1: ml, 2: nlp, 3: tm})
