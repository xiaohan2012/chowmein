import numpy as np


class LabelCountVectorizer(object):
    """
    Count the frequency of labels in each document
    """
    
    def __init__(self):
        pass

    def transform(self, raw_docs, labels):
        """
        Calculate the doc2label frequency table

        Note: docs are not tokenized and frequency is computed
            based on substring matching
        
        raw_docs: list of string
            the document's raw string

        labels: list of string
        """
        ret = np.zeros((len(raw_docs), len(labels)),
                       dtype=np.int32)
        labels = sorted(labels)
        for i, d in enumerate(raw_docs):
            for j, l in enumerate(labels):
                ret[i, j] = d.count(l)
        return ret
        
