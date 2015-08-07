from tmlabel.label_finder import BigramLabelFinder
from tmlabel.data import load_nips
from nose.tools import assert_equal


def test_label_finder():
    finder = BigramLabelFinder(measure='pmi')
    labels = finder.find(load_nips(years=[2009]), 10, top_n=5)
    assert_equal(labels, [(u'monte', u'carlo'),
                          (u'high', u'dimensional'),
                          (u'does', u'not'),  # not so good
                          (u'experimental', u'results'),
                          (u'nonparametric', u'bayesian')])
    
