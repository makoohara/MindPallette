from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, abort
from . import db
from openai import OpenAI
from .models import History
from flask_cors import CORS
from flask_login import current_user, login_required
from datetime import datetime
from collections import Counter
from textblob import TextBlob
import random
import math
import nltk
# resources downloaded
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.corpus import stopwords



from collections import Counter
from dotenv import load_dotenv
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

main = Blueprint('main', __name__)
# Define OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# User credentials and Spotify setup
spotify_client_id = os.getenv("SPOTIFY_CLIENT_ID") 
spotify_client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
spotify_redirect_uri = "http://google.com/callback/"
spotify_scope = "user-read-playback-state,user-modify-playback-state"
diary_stop_words = [
        "I", "me", "my", "mine", "myself",
        "we", "us", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those",
        "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having",
        "do", "does", "did", "doing",
        "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
        "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
        "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
        "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
        "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "s", "today", "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y",
        "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
        "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
        "today", "yesterday", "tomorrow", "day", "night", "morning", "evening"
    ]
class DiaryProcessor:

    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)
        self.diary_stop_words = diary_stop_words

    def analyze_sentiment(self, text):
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        return -1 * math.ceil(abs(polarity) * 5) if polarity < 0 else math.ceil(polarity * 5)

    def extract_nouns(self, text):
        tokens = nltk.word_tokenize(text)
        tags = nltk.pos_tag(tokens)
        #nouns = [word for word, pos in tags if pos in ["NN", "NNS", "NNP", "NNPS"] and word.lower() not in stopwords.words('english')]
        words = [word.lower() for word in tokens if word.lower() not in stopwords.words('english') and word.lower() not in diary_stop_words]
        keywords = [word for word, pos in tags if pos in ['NN', 'NNS', 'JJ', 'JJR', 'JJS']]
        return keywords

        #return [noun for noun in nouns if noun.lower() not in self.diary_stop_words]

    def process_entry_2(self, text):
        print('pipeline 2 text', text)
        # Tokenize the entry
        tokens = word_tokenize(text)
        print('pipeline 2 tokens', tokens)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        print('pipeline 2 filtered_tokens', filtered_tokens)
        # Part-of-Speech Tagging
        tagged_tokens = pos_tag(filtered_tokens)
        print('pipeline 2 tagged_tokens', tagged_tokens)
        # Named Entity Recognition
        named_entities = ne_chunk(tagged_tokens)
        print('pipeline 2 named_entities', named_entities)
        # Extract named entities (people, organizations, locations, etc.)
        themes = [' '.join(c[0] for c in chunk) for chunk in named_entities if hasattr(chunk, 'label')]
        print('pipeline 2 themes', themes)
        # Sentiment Analysis
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(text)
        print('pipeline 2 sentiment', sentiment)
        
        # Extract nouns and adjectives as keywords
        keywords = [word for word, pos in tagged_tokens if pos in ['NN', 'NNS', 'JJ', 'JJR', 'JJS']]
        print('pipeline 2 themes, sentiment, keywords', themes, sentiment, keywords)
        return {
            'themes': themes,
            'sentiment_scores': sentiment,
            'keywords': keywords
        }

    def create_prompt_2(self, prompt_keywords):
        system_msg = f"You are an artist expressing an emotion from a diary into an image using Dalle. You will add artistic interpretation to the prompt. "
        prompt = f"Here are extracted themes, sentiment scores, and keywords form the diary: {prompt_keywords}. Return only the image prompt with this structure: [image type (film, abstract painting, portrait, etc.)] with [description of icon], with [color scheme] and [style]. More specific on art style and format the better."
        print('inside create_prompt_2 prompt', prompt_keywords)
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}],
                temperature=0,
                top_p=1,
                frequency_penalty=0,    
                presence_penalty=0
            )
            prompt = response.choices[0].message.content
            print('pipeline 2 prompt', prompt)
            return prompt
        except Exception as e:
            print(f"Error querying OpenAI for words '{prompt_keywords}': {e}")
            return "unknown"      





    def select_image_attributes(self, scaled_score, nouns):
        # Dictionary of image attributes
        quat_dict = {
            -5: ['muted colors', 'very low saturation', random.choice(['stormy', 'heavy', 'oppressive']), random.choice(['desolation', 'abandonment', 'conflict'])],
            -4: ['cold colors', 'low saturation', random.choice(['gloomy', 'melancholic']), random.choice(['lonely', 'sorrow', 'subtle turmoil'])],
            -3: ['color tone with occasional warm spots', 'moderate saturation', random.choice(['reflective', 'introspective']), random.choice(['nostalgia', 'mild distress', 'uncertainty'])],
            -2: ['balanced colors', 'medium saturation', random.choice(['longing', 'glimmer of hope']), random.choice(['longing'])],
            -1: ['harmonious colors', 'true to life saturation', random.choice(['calm', 'peaceful']), random.choice(['balance', 'everyday life', 'normalcy'])],
            1: ['soft colors', 'slightly high saturation', random.choice(['optimistic', 'fresh']), random.choice(['gentle joy', 'subtle excitement'])],
            2: ['lovely colors', 'high saturation', random.choice(['energetic', 'uplifting']), random.choice(['enthusiasm'])],
            3: ['vivid colors', 'rich saturation', random.choice(['joyful', 'radiant']), random.choice(['happiness', 'fulfillment'])],
            4: ['radiant colors with golden hues', 'extremely high saturation', random.choice(['blissful']), random.choice(['utopic', 'excitement'])],
            5: ['dazzling array of colors', 'extremely high saturation', random.choice(['ethereal', 'transcendent']), random.choice(['utopia', 'paradise'])]
        }
        image_attributes = quat_dict.get(scaled_score, ['neutral'])
        num_nouns = random.randint(1, 3)
        top_nouns = [noun for noun, _ in Counter(nouns).most_common(num_nouns)]
        return image_attributes + top_nouns

    def create_prompt(self, entry):
        scaled_score = self.analyze_sentiment(entry)
        nouns = self.extract_nouns(entry)
        return self.select_image_attributes(scaled_score, nouns)

    def recommend_song(self, entry, genre=None):
        # Building the OpenAI API prompt based on the request body
        prompt = f"Empathize this diary entry with a song: {entry} . Recommend me a song. Return the song name and artist only."

        # Adding genre information to the prompt if available
        genre = request.json.get('genre')
        if genre:
            prompt += f" Pick from the genre = '{genre}'"
        print(prompt)

        # Define the system message
        system_msg = 'You are a helpful assistant who specialize in figurative emotional exploration through songs.'

        song_response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}],
            temperature=0,
            top_p=1,
            frequency_penalty=0,    
            presence_penalty=0
        )
        song_selection = song_response.choices[0].message.content

        return song_selection

    def generate_image_url(self, prompt):
        # Use the GPT API to create a prompt designated for DALL-E
        dalle_prompt = ' '.join(prompt) + ' ,artistic'
        print('final dalle prompt:', dalle_prompt)
        try:
            response = self.client.images.generate(
                        model="dall-e-3",
                        prompt=dalle_prompt,
                        size="1024x1024",
                        quality="standard",
                        n=1,
                        )
            img_url = response.data[0].url
            print('img_url', img_url)
            return img_url
        except Exception as e:
            # Handling errors by sending an error response
            print('OpenAI Dalle Error', str(e))
            return jsonify({'OpenAI Dalle Error': str(e)}), 500

    def process1(self, entry, genre=None):
        song = self.recommend_song(entry, genre)
        image_url = self.generate_image_url(self.create_prompt(entry))
        return {'song': song, 'img_url': image_url}
    
    def process2(self, entry, genre=None):
        print('inside process 2. entry', entry)
        prompt_keywords = self.process_entry_2(entry)
        print('prompt_keywords', prompt_keywords)
        image_url = self.generate_image_url(self.create_prompt_2(prompt_keywords))
        return {'img_url': image_url}


def app_main(request):
    try:
        processor = DiaryProcessor(openai_api_key=OPENAI_API_KEY)
        entry = request.json.get('mood')
        result = processor.process1(entry)
        result2 = processor.process2(entry)
        print('result2', result2)
        return result

    except Exception as e:
        # Handling errors by sending an error response
        return jsonify({'error': str(e)}), 500

def app_2(request):
    try:
        processor = DiaryProcessor(openai_api_key=OPENAI_API_KEY)
        entry = request.json.get('mood')
        result2 = processor.process2(entry)
        print('result2', result2)
        return result2

    except Exception as e:
        # Handling errors by sending an error response
        return jsonify({'error': str(e)}), 500
    


@main.route('/home', methods=['POST', 'GET'])
@login_required
def home():
    if request.method == 'POST':
        data = app_main(request)
        print('data', data)
        #print('data2', app_2(request))
        # Store history
        new_history = History(
            date_time=datetime.utcnow(),
            diary_entry=request.json.get('mood'),
            generated_image=data['img_url'],
            song_snippet=data['song'],
            user_id=current_user.id  # assuming your User model has an id field
        )
        db.session.add(new_history)
        db.session.commit()
        return jsonify(data)
    return render_template('index.html', data=None)


@main.route('/profile')
@login_required
def profile():
    user_history = History.query.filter_by(user_id=current_user.id).order_by(History.date_time.desc()).all()
    return render_template('profile.html', user_name=current_user.name, history=user_history)


@main.route('/delete_history/<int:history_id>')
@login_required
def delete_history(history_id):
    record = History.query.get_or_404(history_id)
    if record.user_id != current_user.id:
        abort(403)  # Forbidden access
    db.session.delete(record)
    db.session.commit()
    flash('Record deleted.', 'success')
    return redirect(url_for('main.profile'))

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

@main.route('/playsong', methods=['POST'])
@login_required
def play_song():
    data = request.json
    song_name = data.get('song')
    song_url = search_and_play_song(song_name)
    if song_url:
        return jsonify({'url': song_url})
    else:
        return jsonify({'error': 'Song not found'}), 404

@main.route('/')
@login_required
def redirect_to_profile():
    return redirect(url_for('main.profile'))