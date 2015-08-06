from nose.tools import (assert_equal, assert_true)
from tmlabel import data


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


