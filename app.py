import os
from flask import Flask, render_template, request, redirect, jsonify, session, url_for
from bs4 import BeautifulSoup
from pre_processor import Normalizer
from data_handler import DataHandler
from algorithm_wrappers import *
from recommender import Recommender
import requests

SONG_DATA = DataHandler.import_pickle("songs_vec_lyrics_set.pk")
TRAINING_DATA = [song.pplyrics for song in SONG_DATA]

GLOVE = GloveWrapper("glove_vectors.txt")
W2V = W2VWrapper()
LDA = LDAWrapper(training_data=TRAINING_DATA)

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.secret_key = 'EUT1820'
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/submit', methods=['GET', 'PUT'])
    def submit():
        if request.method == 'PUT':
            print("Submit pressed")
            data = request.get_json()
            song = data['song']
            artist = data['artist']
            url = data['url']
            current_model_type = data['radio']

            found_in_dataset = False
            for s in SONG_DATA:
                if song.lower() == s.name.lower():
                    lyrics = s.pplyrics
                    found_in_dataset = True
                    break

            if url == "" and found_in_dataset == False:
                lyrics = get_lyrics(song, artist)
                lyrics = Normalizer.full_preprocess(lyrics)
            elif found_in_dataset == False:
                lyrics = get_lyrics(url=url)
                lyrics = Normalizer.full_preprocess(lyrics)

            if lyrics == None:
                return jsonify({"status": "song lyrics not found"}), 404
            else:
                song_vector = DataHandler.calculate_vector(lyrics, GLOVE, W2V, LDA)
                recommender_engine = Recommender(model_type=current_model_type, training_data=SONG_DATA)
                recommended_songs, similarities = recommender_engine.recommend(song_vector[current_model_type], song)
                session['recommendations_song_names'] = [s.name for s in recommended_songs]
                session['recommendations_artists'] = [s.artist for s in recommended_songs]
                session['recommendations_genre'] = [s.mgenre for s in recommended_songs]
                session['recommendations_similarities'] = [str(sim) for sim in similarities]

                return jsonify({"status": "All Good!"}), 200
        
    @app.route("/results", methods=['GET'])    
    def results():
        recommendations_song_names = session.pop('recommendations_song_names', None)
        recommendations_artists = session.pop('recommendations_artists', None)
        recommendations_genre = session.pop('recommendations_genre', None)
        similarities = session.pop('recommendations_similarities', None)
        return render_template('result.html', recommendations_song_names=recommendations_song_names,
                               recommendations_artists=recommendations_artists, 
                               recommendations_genre=recommendations_genre, similarities=similarities)
    
    return app

def get_lyrics(song="", artist="", url=None):
    if url != None:
        page = requests.get(url)
    else:
        query = Normalizer.create_query(str(song), str(artist))
        page = requests.get(query)

    if page.status_code == 200:
        html = BeautifulSoup(page.text, 'html.parser')
        lyrics_containers = html.find_all('div', class_='Lyrics__Container-sc-1ynbvzw-1 kUgSbL')
        all_lyrics = ""
        for container in lyrics_containers:
            lyrics = container.get_text(strip=True)
            all_lyrics += lyrics + "\n"

        return all_lyrics
    else:
        return None

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)