from flask import Blueprint, request, jsonify, abort, url_for
from project import db
from flask_login import current_user, login_required
from datetime import datetime
from .models import History, User
from .main import app_main, OpenAIUtils, NLPUtils, DiaryProcessor
import os
import spotipy

api = Blueprint('api', __name__)
# Define OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
# User credentials and Spotify setup
spotify_client_id = os.getenv("SPOTIFY_CLIENT_ID")
spotify_client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
spotify_redirect_uri = "http://google.com/callback/"
spotify_scope = "user-read-playback-state,user-modify-playback-state"

# @api.route('/api/home', methods=['POST'])
# def home():
#     return jsonify({"status": "success", "message": "Data received!"}), 200


@api.route('/api/home', methods=['POST', 'GET'])
# @login_required
def home():
    if request.method == 'POST':
        data = app_main(request)
        print('data', data)
        return jsonify(data)
    else:
        return jsonify({'message': 'GET method is not supported on this endpoint'}), 405


@api.route('/api/save_image', methods=['POST'])
@login_required
def save_image():
    data = request.json
    img_url = data.get('img_url')
    diary_entry = data.get('diary_entry')
    song = data.get('song')

    new_history = History(
        date_time=datetime.utcnow(),
        diary_entry=diary_entry,
        generated_image=img_url,
        song_snippet=song,
        user_id=current_user.id
    )
    db.session.add(new_history)
    db.session.commit()

    return jsonify({'message': 'Image data saved successfully on db'}), 201


@api.route('/api/profile')
@login_required
def profile():
    user_history = History.query.filter_by(user_id=current_user.id).order_by(History.date_time.desc()).all()
    
    history_list = [
        {
            'id': record.id,
            'date_time': record.date_time.isoformat(),
            'diary_entry': record.diary_entry,
            'generated_image': record.generated_image,
            'song_snippet': record.song_snippet
        }
        for record in user_history
    ]
    
    return jsonify({'user': user_history, 'history': history_list})



@api.route('/api/delete_history/<int:history_id>', methods=['DELETE'])
@login_required
def delete_history(history_id):
    record = History.query.get_or_404(history_id)
    if record.user_id != current_user.id:
        abort(403)  # Forbidden access
    db.session.delete(record)
    db.session.commit()
    return jsonify({'message': 'Record deleted successfully'}), 200


# Create OAuth Object
oauth_object = spotipy.SpotifyOAuth(spotify_client_id, spotify_client_secret, spotify_redirect_uri, scope=spotify_scope)
# Create token
token_dict = oauth_object.get_access_token()
token = token_dict['access_token']
# Create Spotify Object
spotifyObject = spotipy.Spotify(auth=token)


def search_and_play_song(search_keyword):
    # Search for the Song.
    search_results = spotifyObject.search(search_keyword, 1, 0, "track")
    # Get required data from JSON response.
    tracks_dict = search_results['tracks']
    tracks_items = tracks_dict['items']
    if tracks_items:
        return tracks_items[0]['external_urls']['spotify']
    else:
        return None


@api.route('/api/playsong', methods=['POST'])
@login_required
def play_song():
    data = request.json
    song_name = data.get('song')
    try:
        song_url = search_and_play_song(song_name)
        if song_url:
            return jsonify({'url': song_url})
        else:
            return jsonify({'error': 'Song not found'}), 404
    except request.exceptions.ReadTimeout:
        # Handle the timeout, e.g., by logging an error message or notifying the user
        print("The request to Spotify API timed out.")
        return jsonify({'error': 'Request to Spotify API timed out.'}), 504  # Gateway Timeout


@api.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        login_user(user)
        return jsonify({'message': 'Logged in successfully'}), 200
    else:
        return jsonify({'message': 'Invalid credentials'}), 401

@api.route('/api/profile_redirect', methods=['GET'])
@login_required
def api_redirect_to_profile():
    return jsonify({'message': 'Redirect to profile', 'profile_url': url_for('/api/profile')}), 200