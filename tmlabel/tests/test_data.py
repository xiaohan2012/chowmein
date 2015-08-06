from nose.tools import assert_equal
from tmlabel import data


def test_load_nips():
    docs = data.load_nips(years=[2008])
    assert_equal(len(docs), 250)

    docs = data.load_nips()
    assert_equal(len(docs), 2261)
