"""
This class' purpose is to load all the data from the dataset, turn the songs into a class, and export to a .pk file.
Initially the song vectors will be intialised to None, and will be changed later on.

Use this class to read and write to the .pk file
"""
import pandas as pd
import pickle, os, sys
from song import Song
from pre_processor import Normalizer
from algorithm_wrappers import *

LYRICS_PATH = "music4all\music4all_no_audio_csv\lyrics.csv"
GENRES_PATH = "music4all\music4all_no_audio_csv\id_genres.csv"
INFO_PATH = "music4all\music4all_no_audio_csv\id_information.csv"
LANG_PATH = "music4all\music4all_no_audio_csv\id_lang.csv"

class DataHandler():
    def get_lyrics():
        lyrics_df = pd.read_csv(LYRICS_PATH, sep='\t', index_col=0)
        lyrics_df = lyrics_df[lyrics_df.lyrics != "INSTRUMENTAL"]
        return lyrics_df

    def get_genres():
        genres_df = pd.read_csv(GENRES_PATH, sep='\t', index_col=0)
        return genres_df

    def get_info():
        info_df = pd.read_csv(INFO_PATH, sep='\t', index_col=0)
        return info_df

    def get_langs():
        lang_df = pd.read_csv(LANG_PATH, sep='\t', index_col=0)
        lang_df = lang_df[lang_df.lang == "en"]
        return lang_df

    def update_vectors():
        # TODO Do this function when the trainer has been implemented
        return None
    
    def import_pickle(filename):
        with open(filename, 'rb') as pkl:
            return pickle.load(pkl)
    
    def export_pickle(lyrics_df, genres_df, info_df, lang_df, filename):
        all_songs = []

        # We use the lang_df to loop through, since those are the songs we want
        if not os.path.isfile("songs_no_vec_lyrics_set.pk"):
            for i, _ in lang_df.iterrows():
                print(f"PROCESSING: {i}")
                name = info_df.at[i, 'song']
                artist = info_df.at[i, 'artist']
                album = info_df.at[i, 'album_name']
                lyrics = lyrics_df.at[i, 'lyrics']

                genre_list = genres_df.at[i, 'genres'].split(',')
                mgenre = genre_list[0]
                sgenres = genre_list[1:]

                pplyrics = Normalizer.full_preprocess(lyrics)
                vector = None

                # Set pp lyrics and vector to 0 for now
                args = [name, artist, album, lyrics, mgenre, sgenres, pplyrics, vector, i]
                all_songs.append(Song(args))

            # Save what we have so far
            with open("songs_no_vec_lyrics_set.pk", 'wb') as pkl:
                pickle.dump(all_songs, pkl)

        DataHandler.write_vectors(filename)

    def write_vectors(filename):
        all_songs = DataHandler.import_pickle("songs_no_vec_lyrics_set.pk")
        training_data = [song.pplyrics for song in all_songs]
        glove = GloveWrapper("glove_vectors.txt")
        w2v = W2VWrapper()
        lda = LDAWrapper(training_data=training_data)
        for song in all_songs:
            print(f"VECTOR PROCESSING: {song.id}")
            song.vector = DataHandler.calculate_vector(song.pplyrics, glove, w2v, lda)
            if song.vector is None:
                print(f"Song ID: {song.id}, Name: {song.name}")
                sys.exit("Vector is NaN, please retry")

        with open(filename, 'wb') as pkl:
            pickle.dump(all_songs, pkl)

    def prepare_glove(import_filename, export_filename):
        print(f"===Initialising GloVe Export===")
        song_list = DataHandler.import_pickle(import_filename)
        all_lyrics = [song.pplyrics for song in song_list] + [Normalizer.tokenize_lyrics(song.lyrics) for song in song_list]
        with open(export_filename, 'w', encoding='utf-8') as glove_file:
            for pplyrics in all_lyrics:
                lyrics = " ".join(pplyrics)
                glove_file.write(lyrics+"\n")
        print("Done!")

    def update_pickle(song_list, filename):
        with open(filename, 'wb') as pkl:
            pickle.dump(song_list, pkl)
    
    def calculate_vector(pplyrics, glove_model, w2v_model, lda_model):
        """
        Calculates the vectors for a song
        """
        gvectors = []
        wvectors = []

        for token in pplyrics:
            try:
                gvector = glove_model.get_vector(token)
                gvectors.append(gvector)
            except:
                continue

            try:
                wvector = w2v_model.get_vector(token)
                wvectors.append(wvector)
            except:
                continue

        # LDA already creates vector for whole document
        lvector = lda_model.get_vector(pplyrics)

        gvectors = np.vstack(gvectors)
        gvectors_mean = np.mean(gvectors, axis=0)
        if np.all(gvectors_mean == 0):
            print(f"GloVe: Error with vector mean for lyrics {pplyrics}")
            sys.exit("Exiting... Sort it out pls")

        wvectors = np.vstack(wvectors)
        wvectors_mean = np.mean(wvectors, axis=0)
        if np.all(wvectors_mean == 0):
            print(f"W2V: Error with vector mean for lyrics {pplyrics}")
            sys.exit("Exiting... Sort it out pls")

        return {"glove": gvectors_mean, "w2v": wvectors_mean, "lda": lvector}

if __name__ == "__main__":
    if not os.path.isfile("songs_vec_lyrics_set.pk"):
        lyrics_df = DataHandler.get_lyrics()
        genres_df = DataHandler.get_genres()
        info_df = DataHandler.get_info()
        lang_df = DataHandler.get_langs()

        DataHandler.export_pickle(lyrics_df, genres_df, info_df, lang_df, "songs_vec_lyrics_set.pk")

    DataHandler.write_vectors("songs_vec_lyrics_set.pk")


    

    


