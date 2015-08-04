import numpy as np
from label_ranker import LabelRanker


def test_label_ranker():
    ranker = LabelRanker()
    index2word = ('topic', 'modeling', 'machine', 'learning')
    index2label = ('topic modeling', 'machine learning')
    
    topic_models = np.asarray([[0.4, 0.5, 0.05, 0.05],
                              [0.05, 0.05, 0.4, 0.5]])
    pmi_w2l = np.asarray([[1.0, 1.0, 0.2, 0.2],
                          [0.8, 0.8, 0.3, 0.3]
                          [-0.2, 0.2, 0.8, 0.8],
                          [-0.2, 0.0, 0.7, 0.5]])
    ranker.rank(topic_models, pmi_w2l, index2label, index2word, top_n=1)

    
