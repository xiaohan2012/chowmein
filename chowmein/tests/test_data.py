import os
from nose.tools import (assert_equal, assert_true)
from chowmein import data

CURDIR = os.path.dirname(os.path.realpath(__file__))


def test_load_line_corpus():
    docs = data.load_line_corpus(CURDIR + '/../datasets/nips-2008.dat',
                                 tokenize=True)
    assert_equal(len(docs), 250)
    assert_true(isinstance(docs[0], list))

    docs = data.load_line_corpus(CURDIR + '/../datasets/nips-2008.dat',
                                 tokenize=False)
    assert_equal(len(docs), 250)
    assert_true(isinstance(docs[0], basestring))


def test_load_nips_tokenized():
    docs = data.load_nips(years=[2008], raw=False)
    assert_equal(len(docs), 250)
    assert_true(isinstance(docs[0], list))

    docs = data.load_nips(raw=False)
    assert_equal(len(docs), 2261)
    assert_true(isinstance(docs[0], list))


def test_load_nips_raw():
    docs = data.load_nips(years=[2008], raw=True)
    assert_equal(len(docs), 250)
    assert_true(isinstance(docs[0], basestring))

    docs = data.load_nips(raw=True)
    assert_equal(len(docs), 2261)
    assert_true(isinstance(docs[0], basestring))




