import numpy as np
from nose.tools import assert_equal
from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal,
                           assert_almost_equal)
from chowmein.label_ranker import LabelRanker


ranker = LabelRanker()
index2word = ('topic', 'modeling', 'machine', 'learning')
index2label = ('topic modeling', 'machine learning', 'business management')

topic_models = np.asarray([[0.4, 0.5, 0.1, 0.1],
                           [0.05, 0.05, 0.4, 0.5],
                           [0.1, 0.1, 0.1, 0.1]])
pmi_w2l = np.asarray([[1.0, 0.2, 0],
                      [0.8, 0.3, 0.1],
                      [-0.2, 0.8, 0],
                      [0.0, 0.5, 0.1]])
label_models = np.asarray([[0.4, 0.4, 0.1, 0.1],
                           [0.1, 0.2, 0.3, 0.4],
                           [0.25, 0.25, 0.25, 0.25]])


def test_relevance_score():
    label_relevance_score = ranker.label_relevance_score(topic_models,
                                                         pmi_w2l)
    assert_array_almost_equal(label_relevance_score,
                              np.asarray([[0.78, 0.36, 0.06],
                                          [0.01, 0.595, 0.055],
                                          [0.16, 0.18, 0.02]]))


def test_mmr_scores():
    label_scores = ranker.label_relevance_score(topic_models,
                                                pmi_w2l)
    actual_indices, actual_scores\
        = ranker.label_mmr_score(0, [0],  # with 1 label
                                 label_scores,
                                 label_models)
    assert_array_equal(actual_indices, np.asarray([1, 2]))
    alpha = ranker._alpha
    assert_array_almost_equal(actual_scores,
                              np.asarray(
                                  [alpha*0.36 - (1-alpha)*0.606842558,
                                   alpha*0.06 - (1-alpha)*0.223143551]
                              ))


def test_mmr_scores_no_chosen_labels():
    label_scores = ranker.label_relevance_score(topic_models,
                                                pmi_w2l)
    actual_indices, actual_scores\
        = ranker.label_mmr_score(0, [],  # with 0 label
                                 label_scores,
                                 label_models)
    assert_array_equal(actual_indices, np.asarray([0, 1, 2]))
    assert_array_almost_equal(actual_scores,
                              np.asarray(
                                  [0.78, 0.36, 0.06]
                              ))


def test_mmr_scores_max_part():
    # test the `max` part
    label_scores = ranker.label_relevance_score(topic_models,
                                                pmi_w2l)
    actual_indices, actual_scores\
        = ranker.label_mmr_score(0, [0, 1],  # with 2 labels
                                 label_scores,
                                 label_models)
    assert_array_equal(actual_indices, np.asarray([2]))
    alpha = ranker._alpha
    assert_array_almost_equal(actual_scores,
                              np.asarray(
                                  [alpha*0.06 - (1-alpha)*0.223143551]
                              ))


def test_discriminative_scores():
    label_discriminative_score = ranker.label_discriminative_score(
        ranker.label_relevance_score(topic_models, pmi_w2l),
        topic_models,
        pmi_w2l)
    assert_array_almost_equal(label_discriminative_score,
                              np.asarray([[0.085, 0.3875, 0.0375],
                                          [0.47, 0.27, 0.04],
                                          [0.395, 0.4775, 0.0575]]))


def test_combined_label_score():
    label_score = ranker.combined_label_score(topic_models,
                                              pmi_w2l,
                                              use_discrimination=True,
                                              mu=0.7)

    rel_score = np.asarray([[0.78, 0.36, 0.06],
                            [0.01, 0.595, 0.055],
                            [0.16, 0.18, 0.02]])
    dis_score = np.asarray([[0.085, 0.3875, 0.0375],
                            [0.47, 0.27, 0.04],
                            [0.395, 0.4775, 0.0575]])

    assert_almost_equal(label_score[0, 0], 0.7205)
    assert_array_almost_equal(label_score, rel_score - 0.7 * dis_score)

    # without use_discrimination
    label_score = ranker.combined_label_score(topic_models,
                                              pmi_w2l,
                                              use_discrimination=False)
    assert_array_almost_equal(label_score, rel_score)


def test_select_label_sequentially():
    label_score = ranker.combined_label_score(topic_models,
                                              pmi_w2l,
                                              use_discrimination=True,
                                              mu=0.7)
    actual = ranker.select_label_sequentially(1, label_score, label_models)
    expected = np.asarray([[0], [1], [2]])
    assert_array_equal(actual, expected)


def test_select_label_sequentially_all_labels():
    label_score = ranker.combined_label_score(topic_models,
                                              pmi_w2l,
                                              use_discrimination=True,
                                              mu=0.7)
    actual = ranker.select_label_sequentially(3, label_score, label_models)
    expected = np.asarray([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    assert_array_equal(actual, expected)


def test_top_k_labels():
    labels = ranker.top_k_labels(topic_models, pmi_w2l, index2label,
                                 label_models, k=1)
    assert_equal(labels, [['topic modeling'],
                          ['machine learning'],
                          ['business management']])


def test_top_k_labels_all_labels():
    labels = ranker.top_k_labels(topic_models, pmi_w2l, index2label,
                                 label_models, k=3)
    assert_equal(labels, [['topic modeling', 'machine learning',
                           'business management'],
                          ['machine learning', 'business management',
                           'topic modeling'],
                          ['business management', 'topic modeling',
                           'machine learning']])


def test_top_k_labels_coverage_OFF_discrim_OFF():
    ranker = LabelRanker(apply_intra_topic_coverage=False,
                         apply_inter_topic_discrimination=False)
    labels = ranker.top_k_labels(topic_models, pmi_w2l, index2label,
                                 label_models, k=3)
    assert_equal(labels, [['topic modeling', 'machine learning',
                           'business management'],
                          ['machine learning', 'business management',
                           'topic modeling'],
                          ['machine learning',  # because discri OFF
                           'topic modeling',
                           'business management']])


def test_top_k_labels_coverage_OFF_discrim_ON():
    ranker = LabelRanker(apply_intra_topic_coverage=False,
                         apply_inter_topic_discrimination=True)
    labels = ranker.top_k_labels(topic_models, pmi_w2l, index2label,
                                 label_models, k=3)
    assert_equal(labels, [['topic modeling', 'machine learning',
                           'business management'],
                          ['machine learning', 'business management',
                           'topic modeling'],
                          ['business management', 'topic modeling',
                           'machine learning']])


def test_top_k_labels_coverage_ON_discrim_OFF():
    ranker = LabelRanker(apply_intra_topic_coverage=True,
                         apply_inter_topic_discrimination=False)
    labels = ranker.top_k_labels(topic_models, pmi_w2l, index2label,
                                 label_models, k=3)
    assert_equal(labels, [['topic modeling', 'machine learning',
                           'business management'],
                          ['machine learning', 'business management',
                           'topic modeling'],
                          ['machine learning',  # because discri OFF
                           'topic modeling',
                           'business management']])


# NOTE:
# Actually, the power of `intra-coverage` is not demonstrated in this testcase
