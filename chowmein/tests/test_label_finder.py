from chowmein.label_finder import BigramLabelFinder
from chowmein.data import load_nips
from chowmein.corpus_processor import CorpusPOSTagger
from nose.tools import assert_equal


def test_label_finder():
    finder = BigramLabelFinder(measure='pmi', pos=None)
    labels = finder.find(load_nips(years=[2009]), top_n=5)
    assert_equal(labels, [(u'monte', u'carlo'),
                          (u'high', u'dimensional'),
                          (u'does', u'not'),  # not so good
                          (u'experimental', u'results'),
                          (u'nonparametric', u'bayesian')])
    

def test_label_finder_with_pos():
    tagger = CorpusPOSTagger()
    finder = BigramLabelFinder(measure='pmi', pos=[('NN', 'NN'),
                                                   ('JJ', 'NN')])

    docs = load_nips(years=[2009])
    docs = tagger.transform(docs)

    labels = finder.find(docs, top_n=5, strip_tags=False)
    
    assert_equal(labels, [((u'monte', 'NN'), (u'carlo', 'NN')),
                          ((u'nonparametric', 'JJ'), (u'bayesian', 'NN')),
                          ((u'active', 'JJ'), (u'learning', 'NN')),
                          ((u'machine', 'NN'), (u'learning', 'NN')),
                          ((u'semi-supervised', 'JJ'), (u'learning', 'NN'))])

    labels = finder.find(docs, top_n=5)
    
    assert_equal(labels, [(u'monte', u'carlo'),
                          (u'nonparametric', u'bayesian'),
                          (u'active', u'learning'),
                          (u'machine', u'learning'),
                          (u'semi-supervised', u'learning')])
