import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_equal
from chowmein.text import LabelCountVectorizer

ir = 'information retrieval'.split()
ml = 'machine learning'.split()
nlp = 'natural language processing'.split()
tm = 'text mining'.split()

other = ['---']

docs = [ml + other + ml + nlp,
        ir + tm + other + tm,
        ml + other + tm + other,
        nlp + nlp + nlp + ir + other,
        [''],
        other + other + other,
        ml + ml + ml]

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


def test_label_frequency():
    vect = LabelCountVectorizer()
    assert_equal(vect._label_frequency(['a'], ['a', 'a', 'a']), 3)
    assert_equal(vect._label_frequency(['a', 'a'], ['a', 'a', 'a']), 2)
    assert_equal(vect._label_frequency(['a'], []), 0)
    assert_equal(vect._label_frequency(['a'], ['b', 'b', 'b']), 0)
    assert_equal(vect._label_frequency(['a', 'b', 'a'],
                                       ['a', 'b', 'a', 'b', 'a']), 2)
