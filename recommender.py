"""
This class will take a list of songs with their lyrics and perform k-means clustering to provide the recommendations
"""
from algorithm_wrappers import *
from sklearn.cluster import KMeans
from data_handler import *
import numpy as np
from pre_processor import Normalizer

class Recommender():
    def __init__(self, model_type, training_data):
        self.model_type = model_type
        self.training_data = training_data
        self.kmeans_model = self.kmeans()

    def kmeans(self):
        match self.model_type:
            case "w2v":
                vectors = [song.vector["w2v"] for song in self.training_data]
            case "lda":
                vectors = [song.vector["lda"] for song in self.training_data]
            case "glove":
                vectors = [song.vector["glove"] for song in self.training_data]

        vectors = np.vstack(vectors)
        vectors = vectors.astype(np.float32)
        return KMeans(n_clusters=100, random_state=0, n_init="auto").fit(vectors)
    
    def cosine_similarity(self, vec_a, vec_b):
        dot_product = np.dot(vec_a, vec_b)
        magnitude_a = np.linalg.norm(vec_a)
        magnitude_b = np.linalg.norm(vec_b)

        return dot_product / (magnitude_a * magnitude_b)

    def recommend(self, new_song_vector, song_name):
        new_song_vector = new_song_vector.astype(np.float32)
        new_song_label = self.kmeans_model.predict(new_song_vector.reshape(1, -1))
        closest_songs = []
        for i in range(len(self.kmeans_model.labels_)):
            # Each index in the kmeans model corresponds to a song in the dataset
            if self.kmeans_model.labels_[i] == new_song_label and song_name.lower() != self.training_data[i].name.lower():
                closest_songs.append(self.training_data[i])

        # Now onto cosine
        cosine_similarities = [self.cosine_similarity(close_song.vector[self.model_type], new_song_vector) for close_song in closest_songs]
        sorted_indices = np.argsort(cosine_similarities)[::-1][:5]

        sorted_cosines = [cosine_similarities[i] for i in sorted_indices]
        sorted_songs = [closest_songs[i] for i in sorted_indices]
        
        return sorted_songs, sorted_cosines



        

    
