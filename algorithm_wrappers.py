from gensim.test.utils import common_texts, datapath
from gensim.models import Word2Vec, KeyedVectors, LdaModel
from gensim.corpora.dictionary import Dictionary
import numpy as np
import os

"""
This class will initialise a Word2Vec, GloVe or LDA model. The training is done in a separate file to avoid
circular imports.
"""

class W2VWrapper():
    def __init__(self, training_data=None):
        if training_data is None:
            self.word_vectors = KeyedVectors.load("word2vec.wordvectors", mmap='r')
        else:
            print(f"Model not found! Creating...")
            model =  Word2Vec(sentences=training_data, vector_size=50, window=5, min_count=1, workers=4)
            print(f"Word Vectors not found! Creating...")
            model.train(training_data, total_examples=len(training_data), epochs=20)
            model.save("word2vec.model")
            self.word_vectors = model.wv
            self.word_vectors.save("word2vec.wordvectors")

    def update_model(self, lyrics_tokens):
        # We assume we already have a model
        model = Word2Vec.load("word2vec.model")
        model.train(lyrics_tokens, total_examples=len(lyrics_tokens), epochs=15)
        model.save("word2vec.model")
        self.word_vectors = model.wv
        self.word_vectors.save("word2vec.wordvectors")

    def get_vector(self, token):
        return self.word_vectors[token]

class LDAWrapper():
    def __init__(self, training_data, num_topics_param=None):
        # Training data is a list of list of tokens
        self.dictionary = Dictionary(training_data)
        if num_topics_param is None:
            # We assume a model exists and we load it instead of making one
            self.model = LdaModel.load("ldamodel")
        else:
            print(f"Model not found! Creating...")
            training_corpus = [self.dictionary.doc2bow(lyrics) for lyrics in training_data]
            self.model = LdaModel(training_corpus, num_topics=num_topics_param, iterations=40, id2word=self.dictionary)

            self.model.save("ldamodel")

    def update_model(self, lyrics_tokens):
        new_corpus = self.dictionary.doc2bow(lyrics_tokens)
        self.model.update(new_corpus)

    def get_vector(self, lyrics_tokens):
        lyrics_as_corpus = self.dictionary.doc2bow(lyrics_tokens)
        document_topics = self.model.get_document_topics(lyrics_as_corpus, minimum_probability=0, minimum_phi_value=0, per_word_topics=False)
        return np.array([prob for _, prob in document_topics])

class GloveWrapper():
    def __init__(self, filename):
        if not os.path.isfile(filename):
            print(f"Critical failure! GloVe vectors file not found!")
            self.word_vectors = None
        else:
            self.word_vectors = {}
            with open(filename, 'r', encoding='utf-8') as glove_vectors:
                for line in glove_vectors:
                    line_as_list = line.split()
                    key = line_as_list[0]
                    vector_array = np.array([float(vec) for vec in line_as_list[1:]])
                    self.word_vectors[key] = vector_array

    def get_vector(self, token):
        return self.word_vectors[token]



