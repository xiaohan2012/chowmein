import itertools
import codecs
import nltk
from nltk.collocations import BigramCollocationFinder

import pprint

# load data

files = ['nips-2008.dat', 'nips-2009.dat', 'nips-2010.dat',
         'nips-2011.dat', 'nips-2012.dat', 'nips-2013.dat', 'nips-2014.dat']

docs = []
for f in files:
    with codecs.open('datasets/{}'.format(f), 'r', 'utf8') as f:
        doc = []
        for l in f:
            sents = nltk.sent_tokenize(l.strip().lower())
            doc += list(itertools.chain.from_iterable(
                map(nltk.word_tokenize, sents))
            )
        docs.append(doc)

# print docs[:2]

bigram_measures = nltk.collocations.BigramAssocMeasures()

finder = BigramCollocationFinder.from_documents(docs)

finder.apply_freq_filter(10) 

# return the 10 n-grams with the highest PMI
pprint.pprint(finder.nbest(bigram_measures.pmi, 100))


