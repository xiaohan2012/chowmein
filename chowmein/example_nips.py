import lda
import numpy as np

from sklearn.feature_extraction.text import (CountVectorizer
                                             as WordCountVectorizer)
from text import LabelCountVectorizer

from label_finder import BigramLabelFinder
from label_ranker import LabelRanker
from pmi import PMICalculator
from corpus_processor import CorpusWordLengthFilter
from data import (load_nips, load_lemur_stopwords)

n_topics = 6
n_top_words = 15


print("Loading docs...")
docs = load_nips()

print("Word length filtering...")
wl_filter = CorpusWordLengthFilter(minlen=3)
docs = wl_filter.transform(docs)


print("Generate candidate bigram labels...")
finder = BigramLabelFinder('pmi')

cand_labels = finder.find(docs, 10, top_n=200)


print("Calculate the PMI scores...")

pmi_cal = PMICalculator(
    doc2word_vectorizer=WordCountVectorizer(
        min_df=5,
        stop_words=load_lemur_stopwords()),
    doc2label_vectorizer=LabelCountVectorizer())

pmi_w2l = pmi_cal.from_texts(docs, cand_labels)


print("Topic modeling using LDA...")
model = lda.LDA(n_topics=n_topics, n_iter=100,
                # alpha=1.0, eta=1.0,
                random_state=1)
model.fit(pmi_cal.d2w_)

print("Topical words:")
print("-" * 20)
for i, topic_dist in enumerate(model.topic_word_):
    top_word_ids = np.argsort(topic_dist)[:-n_top_words:-1]
    topic_words = [pmi_cal.index2word_[id_]
                   for id_ in top_word_ids]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))


print("Topical labels:")
print("-" * 20)
ranker = LabelRanker(apply_intra_topic_coverage=False)

for i, labels in enumerate(ranker.top_k_labels(
        topic_models=model.topic_word_,
        pmi_w2l=pmi_w2l,
        index2label=pmi_cal.index2label_,
        label_models=None)):

    print "Topic {}: {}".format(
        i,
        ', '.join(map(lambda l: ' '.join(l),
                      labels))
    )
