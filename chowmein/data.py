import os
import nltk
import itertools
import codecs
from toolz.functoolz import compose
import cPickle as pickle

CURDIR = os.path.dirname(os.path.realpath(__file__))


def load_line_corpus(path, tokenize=True):
    docs = []
    with codecs.open(path, "r", "utf8") as f:
        for l in f:
            if tokenize:
                sents = nltk.sent_tokenize(l.strip().lower())
                docs.append(list(itertools.chain(*map(
                    nltk.word_tokenize, sents))))
            else:
                docs.append(l.strip())
    return docs


def load_nips(years=None, raw=False):
    # load data
    if not years:
        years = xrange(2008, 2015)
    files = ['nips-{}.dat'.format(year)
             for year in years]

    docs = []
    for f in files:
        docs += load_line_corpus('{}/datasets/{}'.format(CURDIR, f),
                                 tokenize=(not raw))
        
    return docs


def tag_nips_and_pickle(years=None):
    """
    pos tag the nips collection of given years and then pickle them
    """
    # load data
    if not years:
        years = xrange(2008, 2015)
    files = ['nips-{}.dat'.format(year)
             for year in years]

    for year, f in zip(years, files):

        print("processing year {}...".format(year))

        with codecs.open('{}/datasets/{}'.format(CURDIR, f), 'r', 'utf8') as f:
            docs = []
            for l in f:
                sents = nltk.sent_tokenize(l.strip().lower())
                docs.append(list(itertools.chain(
                    *map(
                        compose(nltk.pos_tag, nltk.word_tokenize),
                        sents))))

            pickle.dump(docs,
                        open('{}/datasets/nips-pos-{}.pkl'.format(CURDIR,
                                                                  year),
                             'w'))
                

def load_tagged_nips(years=None):
    if not years:
        years = xrange(2008, 2015)
        files = ['nips-pos-{}.pkl'.format(year)
                 for year in years]
    docs = []
    for f in files:
        full_path = '{}/datasets/{}'.format(CURDIR, f)
        docs += pickle.load(open(full_path))
    return docs


def load_lemur_stopwords():
    with codecs.open(CURDIR + '/datasets/lemur-stopwords.txt', 
                     'r' 'utf8') as f:
        return map(lambda s: s.strip(),
                   f.readlines())


if __name__ == "__main__":
    tag_nips_and_pickle()
