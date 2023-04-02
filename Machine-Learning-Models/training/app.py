from flask import Flask, render_template, request
import json
import os
import RetrieveSimilarSongs

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    sample_song_path = request.form['sample_song_path']
    config = RetrieveSimilarSongs.Config(
        model_name='short_res',
        batch_size=16,
        model_load_path=RetrieveSimilarSongs.MODEL_LOAD_PATH,
        sample_song_path=sample_song_path,
        songs_path=RetrieveSimilarSongs.DATA_PATH
    )
    s = RetrieveSimilarSongs.RetrieveSimilarSongs(config)
    recommendations = s.give_song_recommendations()
    return json.dumps(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
