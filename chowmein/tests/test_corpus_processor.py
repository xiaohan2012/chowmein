from nose.tools import assert_equal
from chowmein.corpus_processor import (CorpusWordLengthFilter,
                                       CorpusStemmer)


def test_CorpusWordLengthFilter():
    c_filter = CorpusWordLengthFilter(minlen=2, maxlen=5)
    actual = c_filter.transform([['s', 'ok', 'okok', 'too long'],
                                 ['', 'a'*20, 'fine!']])

    expected = [['ok', 'okok'],
                ['fine!']]

    assert_equal(actual, expected)


def test_CorpusStemmer():
    c_stemmer = CorpusStemmer()
    
    actual = c_stemmer.transform(
        [['caresses', 'flies', 'dies', 'mules', 'denied'],
         ['died', 'agreed', 'owned', 'humbled', 'sized']])

    expected = [[u'caress', u'fli', u'die', u'mule', u'deni'],
                [u'die', u'agre', u'own', u'humbl', u'size']]

    assert_equal(actual, expected)


