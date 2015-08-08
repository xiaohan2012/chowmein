import nltk
from nltk.collocations import BigramCollocationFinder
from toolz.itertoolz import get
from toolz.functoolz import partial


class BigramLabelFinder(object):
    def __init__(self, measure='pmi',
                 min_freq=10,
                 pos=[('NN', 'NN'), ('JJ', 'NN')]):
        """
        measure: str
            the measurement method, 'pmi'or 'chi_sq'

        min_freq: int
            minimal frequency for the label to be considered

        pos: list of (str, str)
            the POS tag contraint
        """
        self.bigram_measures = nltk.collocations.BigramAssocMeasures()
        assert measure in ('pmi', 'chi_sq')
        self._measure_method = measure

        self._min_freq = min_freq
        self._pos = pos
        
    def find(self, docs, top_n, strip_tags=True):
        """
        Parameter:
        ---------------

        docs: list of tokenized documents
            
        top_n: int
            how many labels to return

        strip_tags: bool
            whether return without the POS tags or not

        Return:
        ---------------
        list of tuple of str: the bigrams
        """
        # if apply pos constraints
        # check the pos properties
        if self._pos:
            assert isinstance(self._pos, list)
            for pair in self._pos:
                assert isinstance(pair, tuple) or isinstance(pair, list)
                assert len(pair) == 2  # because it's bigram

        score_func = getattr(self.bigram_measures,
                             self._measure_method)

        finder = BigramCollocationFinder.from_documents(docs)
        finder.apply_freq_filter(self._min_freq)

        if self._pos:
            valid_pos_tags = set([pair for pair in self._pos])
            valid_bigrams = []
            bigrams = map(partial(get, 0),  # get the bigram
                          finder.score_ngrams(score_func))
            cnt = 0
            for bigram in bigrams:
                if tuple(map(partial(get, 1), bigram)) in valid_pos_tags:
                    valid_bigrams.append(bigram)
                    cnt += 1
                if cnt == top_n:  # enough
                    break

            if strip_tags:
                valid_bigrams = [tuple(map(partial(get, 0), bigram))
                                 for bigram in valid_bigrams]

            return valid_bigrams
        else:
            bigrams = finder.nbest(score_func,
                                   top_n)
            return bigrams
            
