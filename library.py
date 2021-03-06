import os


import string

import datetime

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

import numpy as np


def get_datetimes(list_of_files):
    """
    Returns a list of datetimes from file names that are stamped with their datetime
    --- E.G. "/some/path/to/year.month.day.ext" would be retrieved and time stamped
    :param list_of_files: The list of file paths -- List[str]
    :return: A numpy array of datetimes -- np.array[datetime]
    """
    base_files = [os.path.basename(f) for f in list_of_files]
    no_ext = [os.path.splitext(f)[0] for f in base_files]
    splits = [f.split('.') for f in no_ext]
    times = np.array(
        [datetime.datetime(int(t[0]), int(t[1]), int(t[2])) for t in splits])
    return times


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


class FileObject:
    def __init__(self):
        self._datetime = None

        self._lines = []

        self._raw = []
        self._stop = []
        self._stem = []
        self._lemma = []

    @property
    def lines(self):
        return self._lines

    def preprocess_text(self):
        self._raw, self._stop, self._stem, self._lemma = preprocess_text(
            self._lines)


def preprocess_text(lines):
    """
    Processes a list of strings, breaking it down into a list of words

    :param lines: The lines to process -- List[str]
    :return: A 4-tuple of pre-processed strings
            (raw, stop word removal, stemmed + stopped, lemma'd + stopped) -- (List[str], ..., ..., ...)
    """
    translator = str.maketrans('', '', string.punctuation)
    upd = []
    for line in lines:
        upd.extend(nltk.sent_tokenize(line))
    lines = [line.translate(translator) for line in upd]
    lines = [nltk.word_tokenize(line) for line in lines]
    lines = [[word.lower() for word in line if word not in [
        '\'', '’', '”', '“']] for line in lines]

    raw = lines

    stop = [[word for word in line if word not in set(
        stopwords.words('english'))] for line in raw]

    snowball_stemmer = SnowballStemmer('english')
    stem = [[snowball_stemmer.stem(word) for word in line] for line in stop]

    wordnet_lemmatizer = WordNetLemmatizer()
    lemma = [[wordnet_lemmatizer.lemmatize(
        word) for word in line] for line in stop]

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
