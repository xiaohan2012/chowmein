from toolz.functoolz import partial
from nltk.stem.porter import PorterStemmer


class CorpusProcessorBase(object):
    """
    Class that processes a corpus
    """
    def transform(self, docs):
        """
        Parameter:
        -----------
        docs: list of (string|list of tokens)
            input corpus
        
        Return:
        ----------
        list of (string|list of tokens):
            transformed corpus
        """
        raise NotImplemented


class CorpusWordLengthFilter(CorpusProcessorBase):
    def __init__(self, minlen=2, maxlen=35):
        self._min = minlen
        self._max = maxlen

    def transform(self, docs):
        """
        Parameters:
        ----------
        docs: list of list of str
            the tokenized corpus
        """
        assert isinstance(docs[0], list)
        valid_length = (lambda word:
                        len(word) >= self._min and
                        len(word) <= self._max)
        filter_tokens = partial(filter, valid_length)
        return map(filter_tokens, docs)
    

porter_stemmer = PorterStemmer()


class CorpusStemmer(CorpusProcessorBase):
    def __init__(self, stemmer=porter_stemmer):
        """
        Parameter:
        --------------
        stemmer: stemmer that accepts list of tokens and stem them
        """
        self._stem = stemmer

    def transform(self, docs):
        """
        Parameter:
        -------------
        docs: list of list of str
            the documents

        Return:
        -------------
        list of list of str: the stemmed corpus
        """
        assert isinstance(docs[0], list)
        stem_tokens = partial(map, self._stem.stem)
        return map(stem_tokens, docs)
