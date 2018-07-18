import os

from collections import Counter
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

import nltk
from datetime import *

from textblob import TextBlob


def get_file_list(text_dir):
    """
    Returns a list of files in a directory

    :param text_dir: The path of the directory -- str
    :return: A list of file paths -- List[str]
    """
    files = []
    for f in os.listdir(text_dir):
        files.append(os.path.join(text_dir, f))
    return files


def preprocess_text(file):
    """
    Loads up a text file and processes it, before breaking it down into a list of words

    :param file: The file path of the text file -- str
    :return: A 4-tuple of pre-processed strings
            (raw, stop word removal, stemmed + stopped, lemma'd + stopped) -- (List[str], ..., ..., ...)
    """
    translator = str.maketrans('', '', string.punctuation)
    with open(file) as in_file:
        lines = in_file.readlines()
        upd = []
        for line in lines:
            upd.extend(nltk.sent_tokenize(line))
        lines = [line.translate(translator) for line in upd]
        lines = [nltk.word_tokenize(line) for line in lines]
        lines = [[word.lower() for word in line if word not in ['\'', '’', '”', '“']] for line in lines]

        raw = lines

        stop = [[word for word in line if word not in set(stopwords.words('english'))] for line in raw]

        snowball_stemmer = SnowballStemmer('english')
        stem = [[snowball_stemmer.stem(word) for word in line] for line in stop]

        wordnet_lemmatizer = WordNetLemmatizer()
        lemma = [[wordnet_lemmatizer.lemmatize(word) for word in line] for line in stop]

        raw = _flatten_list(raw)
        stop = _flatten_list(stop)
        stem = _flatten_list(stem)
        lemma = _flatten_list(lemma)

        return raw, stop, stem, lemma


def preprocess_file_list(file_list):
    """
    Returns a list of word lists from a collection of text files

    :param file_list: The list of file paths to analyze -- List[str]
    :return: A 4-tuple, list of lists of strings -- (List[List[str]], ..., ..., ...)
    """
    raw = []
    stop = []
    stem = []
    lemma = []

    for f in file_list:
        r, s, st, le = preprocess_text(f)
        raw.append(r)
        stop.append(s)
        stem.append(st)
        lemma.append(le)

    return raw, stop, stem, lemma


def _flatten_list(two_list):
    """
    Flattens a two dimensional list

    :param two_list: A two dimensional list -- List[List[any]]
    :return: A one dimensional list -- List[any]
    """
    one_list = []
    for el in two_list:
        one_list.extend(el)
    return one_list


class VocabUtils:

    @staticmethod
    def unique_vocab(word_list):
        """
        Returns a count of the words used in a word list

        :param word_list: A list of words used in a document -- List[str]
        :return: A count of the unique words used in a list -- Counter
        """
        cnt = Counter()
        for word in word_list:
            cnt[word] += 1
        return cnt

    @staticmethod
    def top_k_words(word_list, k=10):
        """
        Returns the top k words used in a list

        :param word_list: A list of words used in a document -- List[str]
        :param k: The top words to be shown -- int
        :return: A list of top words and counts -- List[(str, int)]
        """
        return VocabUtils.unique_vocab(word_list).most_common(k)

    @staticmethod
    def aggregate_words_counts(list_of_word_lists):
        """
        Counts the words used across multiple lists (documents)

        :param list_of_word_lists: A list of list of words used in a collection -- List[List[str]]
        :return: A count of the unique words used in a list -- Counter
        """
        cnt = Counter()
        for word_list in list_of_word_lists:
            cnt += VocabUtils.unique_vocab(word_list)
        return cnt

    @staticmethod
    def global_top_k_words(list_of_word_lists, k=10):
        """
        Returns the top k words used in a collection of word lists (documents)

        :param list_of_word_lists: A list of list of words used in a collection -- List[List[str]]
        :param k: The top words to be shown -- int
        :return: A list of top words and counts -- List[(str, int)]
        """
        return VocabUtils.aggregate_words_counts(list_of_word_lists).most_common(k)

