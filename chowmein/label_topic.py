import argparse
import lda
import numpy as np

from sklearn.feature_extraction.text import (CountVectorizer
                                             as WordCountVectorizer)
from chowmein.text import LabelCountVectorizer
from chowmein.label_finder import BigramLabelFinder
from chowmein.label_ranker import LabelRanker
from chowmein.pmi import PMICalculator
from chowmein.corpus_processor import (CorpusWordLengthFilter,
                                       CorpusPOSTagger,
                                       CorpusStemmer)
from chowmein.data import (load_line_corpus, load_lemur_stopwords)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Command line interface that perform topic modeling " +
        " and topic model labeling")

    # corpus path and preprocessing
    parser.add_argument('--line_corpus_path', type=str, required=True,
                        help="""The path to the corpus.  
                        Each document takes one line.""")

    parser.add_argument('--preprocessing', type=str, nargs='+',
                        default=['wordlen', 'stem', 'tag'],
                        help="""Preprocessing steps to take. Options are:
                        - word length filtering(wordlen) 
                        - stemming(stem),
                        - pos tagging(tag)""")
    # Phrase detection
    parser.add_argument('--n_cand_labels', type=int, default=100,
                        help='Number of candidate labels to take')
    parser.add_argument('--label_tags', type=str, default=['NN,NN', 'JJ,NN'], nargs='+',
                        help="""The POS tag constraint on the candidate labels.
The format is: 
    "{Word 1 tag},{Word 2 tag},...,{Word N tag}"
Multiple constraints can be given.
To disable it, pass 'None'""")
    parser.add_argument('--label_min_df', type=int, default=5,
                        help='Minimum document frequency requirement for candidate labels')

    # LDA
    parser.add_argument('--lda_random_state', type=int, default=12345,
                        help='Random state for LDA modeling')
    parser.add_argument('--lda_n_iter', type=int, default=400,
                        help='Iteraction number for LDA modeling')
    parser.add_argument('--n_topics', type=int, default=6,
                        help='Number of topics')
    parser.add_argument('--n_top_words', type=int, default=15,
                        help='Number of topical words to display for each topic')

    # Topic label
    parser.add_argument('--n_labels', type=int, default=8,
                        help='Number of labels displayed per topic')

    return parser


def get_topic_labels(corpus_path, n_topics,
                     n_top_words,
                     preprocessing_steps,
                     n_cand_labels, label_min_df,
                     label_tags, n_labels,
                     lda_random_state,
                     lda_n_iter):
    """
    Refer the arguments to `create_parser`
    """
    print("Loading docs...")
    docs = load_line_corpus(corpus_path)

    if 'wordlen' in preprocessing_steps:
        print("Word length filtering...")
        wl_filter = CorpusWordLengthFilter(minlen=3)
        docs = wl_filter.transform(docs)

    if 'stem' in preprocessing_steps:
        print("Stemming...")
        stemmer = CorpusStemmer()
        docs = stemmer.transform(docs)

    if 'tag' in preprocessing_steps:
        print("POS tagging...")
        tagger = CorpusPOSTagger()
        tagged_docs = tagger.transform(docs)

    tag_constraints = []
    if label_tags != ['None']:
        for tags in label_tags:
            tag_constraints.append(tuple(map(lambda t: t.strip(),
                                             tags.split(','))))

    if len(tag_constraints) == 0:
        tag_constraints = None

    print("Tag constraints: {}".format(tag_constraints))

    print("Generate candidate bigram labels(with POS filtering)...")
    finder = BigramLabelFinder('pmi', min_freq=label_min_df,
                               pos=tag_constraints)
    if tag_constraints:
        assert 'tag' in preprocessing_steps, \
            'If tag constraint is applied, pos tagging(tag) should be performed'
        cand_labels = finder.find(tagged_docs, top_n=n_cand_labels)
    else:  # if no constraint, then use untagged docs
        cand_labels = finder.find(docs, top_n=n_cand_labels)

    print("Collected {} candidate labels".format(len(cand_labels)))

    print("Calculate the PMI scores...")

    pmi_cal = PMICalculator(
        doc2word_vectorizer=WordCountVectorizer(
            min_df=5,
            stop_words=load_lemur_stopwords()),
        doc2label_vectorizer=LabelCountVectorizer())

    pmi_w2l = pmi_cal.from_texts(docs, cand_labels)

    print("Topic modeling using LDA...")
    model = lda.LDA(n_topics=n_topics, n_iter=lda_n_iter,
                    random_state=lda_random_state)
    model.fit(pmi_cal.d2w_)

    print("\nTopical words:")
    print("-" * 20)
    for i, topic_dist in enumerate(model.topic_word_):
        top_word_ids = np.argsort(topic_dist)[:-n_top_words:-1]
        topic_words = [pmi_cal.index2word_[id_]
                       for id_ in top_word_ids]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    ranker = LabelRanker(apply_intra_topic_coverage=False)

    return ranker.top_k_labels(topic_models=model.topic_word_,
                               pmi_w2l=pmi_w2l,
                               index2label=pmi_cal.index2label_,
                               label_models=None,
                               k=n_labels)
    
if __name__ == '__main__':

    parser = create_parser()

    args = parser.parse_args()
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
    
    print("\nTopical labels:")
    print("-" * 20)
    for i, labels in enumerate(labels):
        print(u"Topic {}: {}\n".format(
            i,
            ', '.join(map(lambda l: ' '.join(l), labels))
        ))

