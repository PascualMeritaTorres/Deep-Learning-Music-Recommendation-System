from flask import Flask, render_template, request, send_from_directory
import json
import os
from werkzeug.utils import secure_filename
from recommendations import RetrieveSimilarSongs
from paths import SHORT_RES_MODEL_PATH,CRNN_MODEL_PATH,MP3_DATA_PATH, MP3_DATA_PATH

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'tmp/'
app.config['SONGS_FOLDER'] = MP3_DATA_PATH

"""
TO-DO: IMPLEMENT A DOWNLOAD BUTTON FOR DJS
"""
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_recommendations', methods=['POST'])
def get_recommendations():
    sample_song = request.files['sample_song_path']
    sample_song_filename = secure_filename(sample_song.filename)
    sample_song_path = os.path.join(app.config['UPLOAD_FOLDER'], sample_song_filename)
    sample_song.save(sample_song_path)

    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    config = Config(
        model_name='short_res',
        batch_size=16,
        short_res_model_load_path=SHORT_RES_MODEL_PATH,
        crnn_model_load_path=CRNN_MODEL_PATH,
        sample_song_path=sample_song_path,
        songs_path=MP3_DATA_PATH
    )
    s = RetrieveSimilarSongs(config)
    recommendations = s.give_song_recommendations()

    # Add song URLs to recommendations
    for song in recommendations:
        song["song_url"] = f"/songs/{song['song_id']}"

    # Remove the temporary file after processing.
    os.remove(sample_song_path)

    return json.dumps(recommendations)

@app.route('/songs/<path:song_id>')
def serve_song(song_id):
    return send_from_directory(app.config['SONGS_FOLDER'], song_id)

if __name__ == '__main__':
    app.run(debug=True)
