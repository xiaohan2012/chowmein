from tmlabel.label_finder import BigramLabelFinder
from tmlabel.data import load_nips
from nose.tools import assert_equal


def test_label_finder():
    finder = BigramLabelFinder(measure='pmi')
    labels = finder.find(load_nips(years=[2009]), 10, top_n=5)
    assert_equal(labels, [u'monte carlo',
                          u'high dimensional',
                          u'does not',  # not so good
                          u'experimental results',
                          u'nonparametric bayesian'])
    
