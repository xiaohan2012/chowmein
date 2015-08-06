import os
import nltk
import itertools
import codecs

CURDIR = os.path.dirname(os.path.realpath(__file__))


def load_nips(years=None):
    # load data
    if not years:
        years = xrange(2008, 2015)
    files = ['nips-{}.dat'.format(year)
             for year in years]

    docs = []
    for f in files:
        with codecs.open('{}/datasets/{}'.format(CURDIR, f), 'r', 'utf8') as f:
            for l in f:
                sents = nltk.sent_tokenize(l.strip().lower())
                docs.append(list(itertools.chain(*map(
                    nltk.word_tokenize, sents))))
    return docs
