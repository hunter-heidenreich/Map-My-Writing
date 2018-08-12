from allennlp.commands.elmo import ElmoEmbedder

import numpy as np
import os

from keras.datasets import imdb
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.preprocessing import sequence
from keras.models import Sequential


from sigopt import Connection

import fastText


class SentimentAnalyzer:
    def __init__(self, opt):
        self.opt = opt
        self.ft = fastText.load_model(
            '/Users/hunterheidenreich/git/MapMyWriting/crawl-300d-2M.vec')

        top_words = self.opt['top_words']
        (self.X_train, self.y_train), \
            (self.X_test, self.y_test) = imdb.load_data(num_words=top_words)

        max_words = self.opt['max_words']
        self.X_train = sequence.pad_sequences(
            self.X_train, maxlen=max_words)
        self.X_test = sequence.pad_sequences(
            self.X_test, maxlen=max_words)

        self.classifier = self._create_classifier()

    def train_classifier(self):
        return self._train_classifier()

    def evaluate_classifier(self):
        X, y = self._get_validation_set()
        return self.classifier.evaluate(X, y, batch_size=1)

    def _create_classifier(self):
        model = Sequential()
        model.add(Conv1D(filters=np.power(2, self.opt['filters']),
                         kernel_size=self.opt['kernel_size'],
                         padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=self.opt['pool_size']))
        model.add(Flatten())
        model.add(Dense(self.opt['dense_size'], activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        return model

    def _data_generator(self):
        for i in range(len(self.X_train // self.opt['batch_size'])):
            X_sample = self.X_train[i * self.opt['batch_size']
                :i * self.opt['batch_size'] + self.opt['batch_size']]
            y_sample = self.y_train[i * self.opt['batch_size']
                :i * self.opt['batch_size'] + self.opt['batch_size']]

            X_sample = [[self.ft.get_word_vector(
                word) for word in fastText.tokenize(sent)]for sent in X_sample]

            yield X_sample, y_sample

    def _get_validation_set(self):
        p = np.random.permutation(len(self.X_test))
        self.X_test = self.X_test[p]
        self.y_test = self.y_test[p]

        X_sample = self.X_test[:100]
        y_sample = self.y_test[:100]

        X_sample = [[self.ft.get_word_vector(
            word) for word in fastText.tokenize(sent)] for sent in X_sample]
        return X_sample, y_sample

    def _train_classifier(self):
        return self.classifier.fit_generator(self._data_generator(), epochs=3)


def run_sigopt_optimization():
    conn = Connection(
        client_token=os.environ['SIGOPT_TOKEN'])
    # experiment = conn.experiments().create(
    #     name='Simple CNN Sentiment Analysis Hyperparameter Search',
    #     parameters=[
    #         dict(name='top_words', type='int', bounds=dict(min=100, max=88000)),
    #         dict(name='max_words', type='int', bounds=dict(min=10, max=1000)),
    #         dict(name='filters', type='int', bounds=dict(min=0, max=10)),
    #         dict(name='kernel_size', type='int', bounds=dict(min=1, max=5)),
    #         dict(name='pool_size', type='int', bounds=dict(min=2, max=8)),
    #         dict(name='dense_size', type='int', bounds=dict(min=2, max=1024)),
    #         dict(name='batch_size', type='int', bounds=dict(min=2, max=1024)),
    #     ],
    # )
    # print("Created experiment: https://sigopt.com/experiment/" + experiment.id)
    e_id = str(48737)
    for _ in range(300):
        suggestion = conn.experiments(e_id).suggestions().create()

        options = suggestion.assignments

        s = SentimentAnalyzer(options)
        s.train_classifier()
        value = s.evaluate_classifier()

        conn.experiments(e_id).observations().create(
            suggestion=suggestion.id,
            value=value[1],
        )


if __name__ == '__main__':
    run_sigopt_optimization()
