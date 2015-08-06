import nltk
from nltk.collocations import BigramCollocationFinder


class BigramLabelFinder(object):
    def __init__(self, measure='pmi'):
        """
        measure: str
            the measurement method, 'pmi'or 'chi_sq'
        """
        self.bigram_measures = nltk.collocations.BigramAssocMeasures()
        assert measure in ('pmi', 'chi_sq')
        self._measure_method = measure
    
    def find(self, docs, min_freq, top_n):
        """
        Parameter:
        ---------------

        docs: list of tokenized documents
            
        min_freq: int
            the minimal frequency to be considered

        top_n: int
            how many labels to return

        Return:
        ---------------
        list of str: the bigrams
        """
        
        finder = BigramCollocationFinder.from_documents(docs)
        finder.apply_freq_filter(min_freq)
        bigrams = finder.nbest(getattr(self.bigram_measures,
                                       self._measure_method),
                               top_n)
        labels = []
        for b in bigrams:
            labels.append(' '.join(b))

        return labels
