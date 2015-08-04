"""
Reference:
---------------------

Qiaozhu Mei, Xuehua Shen, Chengxiang Zhai,
Automatic Labeling of Multinomial Topic Models, 2007
"""


class LabelRanker(object):
    """
    
    """
    def __init__(self,
                 intra_topic_coverage=True,
                 inter_topic_discrimination=True,
                 mu=0.7,
                 alpha=1.0):
        self._coverage = intra_topic_coverage
        self._discrimination = inter_topic_discrimination
        self._mu = mu
        self._alpha = alpha
        
    def rank(self,
             topic_models,
             pmi_w2l,
             index2label,
             index2word,
             top_n=5
             ):
        """
        Parameters:
        ---------------
        topic_models: numpy.ndarray(#topics, #words)
           the topic models

        pmi_w2l: numpy.ndarray(#words, #labels)
           the Point-wise Mutual Information(PMI) table of
           the form, PMI(w, l | C)
        
        index2label: dict<int, object>
           mapping from label index in the `pmi_w2l`
           to the label object, which can be string

        index2word: dict<int, object>
           mapping from label index in `topic_models`
           to the word, which is often string

        top_n: in
           how many labels returned for each topic model
        
        Returns;
        -------------
        list<list of <label, float>>
           #top_n labels as well as scores for each topic model
        
        """
        
