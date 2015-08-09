import os
from nose.tools import assert_equal
from chowmein.label_topic import (get_topic_labels, create_parser)

CURDIR = os.path.dirname(os.path.realpath(__file__))
data_path = CURDIR + '/../datasets/nips-2014.dat'

parser = create_parser()


def test_get_topc_labels():
    # test if it's running and returns the correct number of topical labels
    args = parser.parse_args(['--line_corpus_path', data_path,
                              '--preprocessing', 'wordlen', 'tag',
                              '--label_tags', 'NN,NN', 'JJ,NN',
                              '--n_cand_labels', '200',
                              '--n_labels', '5',
                              '--n_topics', '2'])

    labels = get_topic_labels(corpus_path=args.line_corpus_path,
                              n_topics=args.n_topics,
                              n_top_words=args.n_top_words,
                              preprocessing_steps=args.preprocessing,
                              n_cand_labels=args.n_cand_labels,
                              label_min_df=args.label_min_df,
                              label_tags=args.label_tags,
                              n_labels=args.n_labels,
                              lda_random_state=args.lda_random_state,
                              lda_n_iter=args.lda_n_iter)
    assert_equal(len(labels), 2)
    for l in labels:
        assert_equal(len(l), 5)
