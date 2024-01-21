#!/bin/bash

from data_handler import *
from algorithm_wrappers import *
from pre_processor import Normalizer
import subprocess

def retrain():
    # Function if I need to retrain EVERYTHING
    all_lyrics_tokens = [song.pplyrics for song in DataHandler.import_pickle("songs_no_vec_lyrics_set.pk")] + [Normalizer.tokenize_lyrics(song.lyrics) for song in DataHandler.import_pickle("songs_no_vec_lyrics_set.pk")]

    # Gonna have to do GloVe manually bud

    #Word2Vec
    print("Training Word2Vec")
    w2v = W2VWrapper(training_data=all_lyrics_tokens)

    #LDA
    print("Training LDA")
    lda = LDAWrapper(num_topics_param=100, training_data=all_lyrics_tokens)


if __name__ == '__main__':
    retrain()