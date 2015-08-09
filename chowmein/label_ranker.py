"""
Reference:
---------------------

Qiaozhu Mei, Xuehua Shen, Chengxiang Zhai,
Automatic Labeling of Multinomial Topic Models, 2007
"""
import numpy as np
from scipy.stats import entropy as kl_divergence


class LabelRanker(object):
    """
    
    """
    def __init__(self,
                 apply_intra_topic_coverage=True,
                 apply_inter_topic_discrimination=True,
                 mu=0.7,
                 alpha=0.9):
        self._coverage = apply_intra_topic_coverage
        self._discrimination = apply_inter_topic_discrimination
        self._mu = mu
        self._alpha = alpha

    def label_relevance_score(self,
                              topic_models,
                              pmi_w2l):
        """
        Calculate the relevance scores between each label and each topic

        Parameters:
        ---------------
        topic_models: numpy.ndarray(#topics, #words)
           the topic models

        pmi_w2l: numpy.ndarray(#words, #labels)
           the Point-wise Mutual Information(PMI) table of
           the form, PMI(w, l | C)
        
        Returns;
        -------------
        numpy.ndarray, shape (#topics, #labels)
            the scores of each label on each topic
        """
        assert topic_models.shape[1] == pmi_w2l.shape[0]
        return np.asarray(np.asmatrix(topic_models) *
                          np.asmatrix(pmi_w2l))
        
    def label_discriminative_score(self,
                                   relevance_score,
                                   topic_models,
                                   pmi_w2l):
        """
        Calculate the discriminative scores for each label
        
        Returns:
        --------------
        numpy.ndarray, shape (#topics, #labels)
            the (i, j)th element denotes the score
            for label j and all topics *except* the ith
        """
        assert topic_models.shape[1] == pmi_w2l.shape[0]
        k = topic_models.shape[0]
        return (relevance_score.sum(axis=0)[None, :].repeat(repeats=k, axis=0)
                - relevance_score) / (k-1)
        
    def label_mmr_score(self,
                        which_topic,
                        chosen_labels,
                        label_scores,
                        label_models):
        """
        Maximal Marginal Relevance score for labels.
        It's computed only when `apply_intra_topic_coverage` is True

        Parameters:
        --------------
        which_topic: int
            the index of the topic
        
        chosen_labels: list<int>
           indices of labels that are already chosen
        
        label_scores: numpy.ndarray<#topic, #label>
           label scores for each topic

        label_models: numpy.ndarray<#label, #words>
            the language models for labels

        Returns:
        --------------
        numpy.ndarray: 1D of length #label - #chosen_labels
            the scored label indices

        numpy.ndarray: same length as above
            the scores
        """
        chosen_len = len(chosen_labels)
        if chosen_len == 0:
            # no label is chosen
            # return the raw scores
            return (np.arange(label_models.shape[0]),
                    label_scores[which_topic, :])
        else:
            kl_m = np.zeros((label_models.shape[0]-chosen_len,
                             chosen_len))
            
            # the unchosen label indices
            candidate_labels = list(set(range(label_models.shape[0])) -
                                    set(chosen_labels))
            candidate_labels = np.sort(np.asarray(candidate_labels))
            for i, l_p in enumerate(candidate_labels):
                for j, l in enumerate(chosen_labels):
                    kl_m[i, j] = kl_divergence(label_models[l_p],
                                               label_models[l])
            sim_scores = kl_m.max(axis=1)
            mml_scores = (self._alpha *
                          label_scores[which_topic, candidate_labels]
                          - (1 - self._alpha) * sim_scores)
            return (candidate_labels, mml_scores)

    def combined_label_score(self, topic_models, pmi_w2l,
                             use_discrimination, mu=None):
        """
        Calculate the combined scores from relevance_score
        and discrimination_score(if required)

        Parameter:
        -----------
        use_discrimination: bool
            whether use discrimination or not
        mu: float
            the `mu` parameter in the algorithm

        Return:
        -----------
        numpy.ndarray, shape (#topics, #labels)
            score for each topic and label pair
        """
        rel_scores = self.label_relevance_score(topic_models, pmi_w2l)
        
        if use_discrimination:
            assert mu != None
            discrim_scores = self.label_discriminative_score(rel_scores,
                                                             topic_models,
                                                             pmi_w2l)
            label_scores = rel_scores - mu * discrim_scores
        else:
            label_scores = rel_scores

        return label_scores

    def select_label_sequentially(self, k_labels,
                                  label_scores, label_models):
        """
        Return:
        ------------
        list<list<int>>: shape n_topics x k_labels
        """
        n_topics = label_scores.shape[0]
        chosen_labels = []

        # don't use [[]] * n_topics !
        for _ in xrange(n_topics):
            chosen_labels.append(list())
            
        for i in xrange(n_topics):
            for j in xrange(k_labels):
                inds, scores = self.label_mmr_score(i, chosen_labels[i],
                                                    label_scores,
                                                    label_models)
                chosen_labels[i].append(inds[np.argmax(scores)])
        return chosen_labels

    def top_k_labels(self,
                     topic_models,
                     pmi_w2l,
                     index2label,
                     label_models=None,
                     k=5):
        """
        Parameters:
        ----------------
        
        index2label: dict<int, object>
           mapping from label index in the `pmi_w2l`
           to the label object, which can be string

        label_models: numpy.ndarray<#label, #words>
            the language models for labels
            if `apply_intra_topic_coverage` is True,
            then it's must be given

        Return:
        ---------------
        list<list of (label, float)>
           top k labels as well as scores for each topic model

        """

        assert pmi_w2l.shape[1] == len(index2label)

        label_scores = self.combined_label_score(topic_models, pmi_w2l,
                                                 self._discrimination,
                                                 self._mu)

        if self._coverage:
            assert isinstance(label_models, np.ndarray)
            # TODO: can be parallel
            chosen_labels = self.select_label_sequentially(k, label_scores,
                                                           label_models)
        else:
            chosen_labels = np.argsort(label_scores, axis=1)[:, :-k-1:-1]
        return [[index2label[j]
                 for j in topic_i_labels]
                for topic_i_labels in chosen_labels]
            
    def print_top_k_labels(self, topic_models, pmi_w2l,
                           index2label, label_models, k):
        res = u"Topic labels:\n"
        for i, labels in enumerate(self.top_k_labels(
                topic_models=topic_models,
                pmi_w2l=pmi_w2l,
                index2label=index2label,
                label_models=label_models,
                k=k)):
            res += u"Topic {}: {}\n".format(
                i,
                ', '.join(map(lambda l: ' '.join(l),
                              labels))
            )
        return res
