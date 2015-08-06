from scipy.sparse import csr_matrix
from scipy import int64


class LabelCountVectorizer(object):
    """
    Count the frequency of labels in each document
    """
    
    def __init__(self):
        self.index2label_ = None
        
    def transform(self, raw_docs, labels):
        """
        Calculate the doc2label frequency table

        Note: docs are not tokenized and frequency is computed
            based on substring matching
        
        Parameter:
        ------------

        raw_docs: list of string
            the document's raw string

        labels: list of string

        Return:
        -----------
        scipy.sparse.csr_matrix: #doc x #label
            the frequency table
        """
        labels = sorted(labels)
        self.index2label_ = {index: label
                             for index, label in enumerate(labels)}

        ret = csr_matrix((len(raw_docs), len(labels)),
                         dtype=int64)
        for i, d in enumerate(raw_docs):
            for j, l in enumerate(labels):
                cnt = d.count(l)
                if cnt > 0:
                    ret[i, j] = d.count(l)
        return ret
        
