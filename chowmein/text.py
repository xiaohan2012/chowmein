from scipy.sparse import (csr_matrix, lil_matrix)
from scipy import int64


class LabelCountVectorizer(object):
    """
    Count the frequency of labels in each document
    """
    
    def __init__(self):
        self.index2label_ = None
        
    def _label_frequency(self, label_tokens, context_tokens):
        """
        Calculate the frequency that the label appears
        in the context(e.g, sentence)
        
        Parameter:
        ---------------

        label_tokens: list|tuple of str
            the label tokens
        context_tokens: list|tuple of str
            the sentence tokens

        Return:
        -----------
        int: the label frequency in the sentence
        """
        label_len = len(label_tokens)
        cnt = 0
        for i in xrange(len(context_tokens) - label_len + 1):
            match = True
            for j in xrange(label_len):
                if label_tokens[j] != context_tokens[i+j]:
                    match = False
                    break
            if match:
                cnt += 1
        return cnt

    def transform(self, docs, labels):
        """
        Calculate the doc2label frequency table

        Note: docs are not tokenized and frequency is computed
            based on substring matching
        
        Parameter:
        ------------

        docs: list of list of string
            tokenized documents

        labels: list of list of string

        Return:
        -----------
        scipy.sparse.csr_matrix: #doc x #label
            the frequency table
        """
        labels = sorted(labels)
        self.index2label_ = {index: label
                             for index, label in enumerate(labels)}

        ret = lil_matrix((len(docs), len(labels)),
                         dtype=int64)
        for i, d in enumerate(docs):
            for j, l in enumerate(labels):
                cnt = self._label_frequency(l, d)
                if cnt > 0:
                    ret[i, j] = cnt
        return ret.tocsr()

