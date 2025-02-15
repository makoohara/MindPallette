from flask import Flask, Blueprint, request, jsonify, url_for, abort
from . import db
from openai import OpenAI
from .models import History
from flask_login import current_user, login_required
from datetime import datetime
import random
import math
import nltk
import re
from itertools import islice
import numpy as np
from dotenv import load_dotenv
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from allennlp.predictors.predictor import Predictor
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('words')


load_dotenv()

main = Blueprint('main', __name__)
# Define OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
# User credentials and Spotify setup
spotify_client_id = os.getenv("SPOTIPY_CLIENT_ID")
spotify_client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
spotify_redirect_uri = "http://google.com/callback/"
spotify_scope = "user-read-playback-state,user-modify-playback-state"

class DiaryProcessor:

    def __init__(self, openai_util, nlp_util):
        self.openai_util = openai_util
        self.nlp_util = nlp_util

    def select_image_attributes(self, scaled_score):
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
        return image_attributes

    def data_process(self, entry):
        tokenized_sentences = self.nlp_util.tokenize(entry)
        polarity = self.nlp_util.sentiment_analysis(tokenized_sentences)

        main_icons, sub_icons = self.nlp_util.icon_extraction(entry)
        sentiment_stats = self.nlp_util.sentiment_stats(polarity)
        overall_sentiment = sentiment_stats['overall_sentiment']
        print('overall_sentiment:', overall_sentiment)
        scaled_score = -1 * math.ceil(abs(overall_sentiment) * 5) if overall_sentiment < 0 else math.ceil(overall_sentiment * 5)
        print('scaled_score:', scaled_score)
        srl_result = self.nlp_util.srl(tokenized_sentences)
        print('srl_result:', srl_result)
        annotations = self.nlp_util.annotate_srl(srl_result)
        print('annotations:', annotations)
        icons = {'main_icons': main_icons, "sub_icons": sub_icons}
        data = {
            1: {**icons, **{'image_attributes': self.select_image_attributes(scaled_score)}},
            2: {'keywords': icons, 'overall sentiment': overall_sentiment},
            'normalized overall std': sentiment_stats['normalized_overall_std'],
            3: {'text': entry},
            4: {'annotations': annotations, 'sentiments': sentiment_stats}
        }
        return data


    def output_cleaning(self, prompt):
        pattern = r"\{[^{}]*\}"
        match = re.search(pattern, prompt)
        if match:
            dict_str = match.group(0)
            # Assuming the extracted string is safe to evaluate
            parameters = eval(dict_str)
        return parameters


class NLPUtils:
    def __init__(self):
        self.diary_stop_words = [
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
        "today", "yesterday", "tomorrow", "day", "night", "morning", "evening"]
        self.model_path = {'srl': "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz",
                            'coref': "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"}

    def icon_extraction(self, entry):
        COREF_MODEL_PATH = self.model_path['coref']
        predictor = Predictor.from_path(COREF_MODEL_PATH)
        predictions = predictor.predict(document=entry)
        document = predictions['document']
        icons = dict()
        clusters = predictions['clusters']
        for cluster in clusters:
            freq = len(cluster)
            cluster_index = cluster[0]
            word = " ".join(document[cluster_index[0]:cluster_index[1]+1])
            if word in self.diary_stop_words:
                continue
            icons[word] = freq
        sorted_icons_desc = {word: freq for word, freq in sorted(icons.items(), key=lambda item: item[1], reverse=True)}
        # make the abstraction random
        main_keyword_index = random.randint(1, 3)
        sub_keyword_index = main_keyword_index + random.randint(1, 3)
        main_icons = dict(islice(sorted_icons_desc.items(), 0, main_keyword_index))
        sub_icons = dict(islice(sorted_icons_desc.items(), main_keyword_index, sub_keyword_index))
        print('main_icons:', main_icons, 'sub_icons:', sub_icons)
        return main_icons, sub_icons

    def tokenize(self, entry):
        sentences = sent_tokenize(entry)
        return sentences


    def srl(self, tokenized_sentences):
        """
        Extract verbs from a sentence using SRL, then query ChatGPT for associated emotions.
        """
        srl_model_path = self.model_path['srl']
        predictor = Predictor.from_path(srl_model_path)
        result = []
        for sentence in tokenized_sentences:
            srl_result = predictor.predict(sentence=sentence)
            result.append(srl_result)
        return result

    def annotate_srl(self, srl_result):
        structured_data = {}
        for sentence in srl_result:
            verbs = sentence['verbs']
            for verb_dict in verbs:
                roles = verb_dict['description']
                components = re.findall(r'\[(.*?)\]', roles)
                for component in components:
                    # Split each component into its label and text
                    label, text = component.split(': ', 1)
                    if label in structured_data:
                        structured_data[label] += [text]
                    else: 
                        structured_data[label] = [text]
        return structured_data

    def sentiment_analysis(self, tokenized_sentences):
        sia = SentimentIntensityAnalyzer()
        compound = []
        pos = []
        neg = []

        for sentence in tokenized_sentences:
            sentiment = sia.polarity_scores(sentence)
            compound.append(sentiment['compound'])
            pos.append(sentiment['pos'])
            neg.append(sentiment['neg'])
        sentiments = {'compound': compound, 'positive': pos, 'negative': neg}
        return sentiments

    def sentiment_stats(self, sentiment_analysis):
        overall_sentiment = np.mean(sentiment_analysis['compound'])
        # cv = sv/mu
        normalized_overall_std = np.std(sentiment_analysis['compound'])/overall_sentiment
        max_sentiment = np.max(sentiment_analysis['compound'])
        min_sentiment = np.min(sentiment_analysis['compound'])

        # Calculate the proportion of positive to negative sentiments
        positive_mean = np.mean(sentiment_analysis['positive'])
        negative_mean = np.mean(sentiment_analysis['negative'])
        sentiment_ratio = positive_mean / negative_mean if negative_mean != 0 else float('inf')

        # Quantify sentiment polarity (number of positive, neutral, and negative compound scores)
        positive_count = len([score for score in sentiment_analysis['compound'] if score > 0])
        neutral_count = len([score for score in sentiment_analysis['compound'] if score == 0])
        negative_count = len([score for score in sentiment_analysis['compound'] if score < 0])

        # Prepare the output
        output = {
            'overall_sentiment': overall_sentiment,
            'normalized_overall_std': normalized_overall_std,
            'max_sentiment': max_sentiment,
            'min_sentiment': min_sentiment,
            'sentiment_ratio': sentiment_ratio,
            'positive_count': positive_count,
            'neutral_count': neutral_count,
            'negative_count': negative_count
        }
        return output


class OpenAIUtils:
    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)
    def query(self, system_msg, prompt):
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            parameters = response.choices[0].message.content
            return parameters
        except Exception as e:
            # Handling errors by sending an error response
            print('OpenAI GPT-3.5 Error', str(e))
            return jsonify({'OpenAI GPT-3.5 Error': str(e)}), 500

    def generate_parameters(self, processed_data, pipeline):
        system_msg = 'You are a prompt engineer for Dalle. You improvise or adjust prompts based on annotated/original text inputs and returns parameter as an python object. The goal of the prompt is to visually and figuratively express the emotion in the diary. The more detailed and nuanced the better.'
        parameters = {
            'main subjects': {'London': 1},
            'secondary subjects': {'orange': 1},
            'aspect ratio': '1:1',
            'medium': 'abstract painting',
            'camera': {'portrait view': 1},
            'descriptor': {'realism': 1},
            'artist': {'Andy Warhol': 1},
            'lighting': {'black light': 1},
            'color': {'dark': 1}
        }
        if pipeline == 1:
            main_icons = processed_data['main_icons']
            sub_icons = processed_data['sub_icons']
            attributes = processed_data['image_attributes']
            prompt = f"main subjects: {main_icons}, secondary subjects: {sub_icons}, attributes: {attributes}"
            return prompt
        elif pipeline == 2:
            prompt_keywords = processed_data['keywords']
            prompt = f"Here are extracted themes, sentiment scores, and keywords from the diary: {prompt_keywords}. Return only the image prompt for Dall-E with this structure: [image type (e.g. film, abstract painting, portrait, etc.)] of [description of icon], with [color scheme] and [style/artist]. More specific on art style/detail the better."
        elif pipeline == 3:
            text = processed_data['text']
            prompt = f"Return parameters for a Dalle prompt based on a diary. The parameter structure should look like this: {parameters}. \n\
                        Your task is to change the item values of the parameter to match this diary text. {text} "
        elif pipeline == 4:
            annotations = processed_data['annotations']
            sentiments = processed_data['sentiments']
            prompt = f"Return parameters for a Dalle prompt based on a diary. The parameter structure should look like this: {parameters}. \n\
                        Your task is to change the item values of the parameter dictionary to match this annotated text and sentiment states from a diary here. {annotations}, {sentiments} \n\
                        Do not put any artist names."
     
        try:
            print('Dalle prompt', pipeline, ":", prompt)
            return self.query(system_msg, prompt)
        except Exception as e:
            # Handling errors by sending an error response
            print('OpenAI GPT-3.5 Error', str(e))
            return jsonify({'OpenAI GPT-3.5 Error': str(e)}), 500
        
    def generate_image_url(self, prompt):
        dalle_prompt = ' '.join(prompt) + ' ,artistic'
        try:
            response = self.client.images.generate(
                        model="dall-e-3",
                        prompt=dalle_prompt,
                        size="1024x1024",
                        quality="standard",
                        n=1,
                        )
            img_url = response.data[0].url
            return img_url
        except Exception as e:
            # Handling errors by sending an error response
            print('OpenAI Dalle Error', str(e))
            return jsonify({'OpenAI Dalle Error': str(e)}), 500


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


def app_main(request):
    try:

        openai_util = OpenAIUtils(openai_api_key=openai_api_key)
        nlp_util = NLPUtils()
        processor = DiaryProcessor(openai_util, nlp_util)
        entry = request.json.get('mood')
        processed_data = processor.data_process(entry)

        results = []
        for i in range(1, 4):
            pipeline = i
            data = processed_data[i]
            print('pipeline', i, data)
            
            prompt = openai_util.generate_parameters(data, pipeline)

            if i == 3:
                prompt = processor.output_cleaning(prompt)

            print('prompt', i, prompt, type(prompt))
            result = openai_util.generate_image_url(prompt)

            print('result', i, result)
            
            results.append(str(result))

        song = openai_util.recommend_song(entry)
        print("song object", song, type(song))
        return {'song': song, 'img_urls': results}

    except Exception as e:
        # Handling errors by sending an error response
        return jsonify({'error': str(e)}), 500


@main.route('/api/home', methods=['POST', 'GET'])
# @login_required
def home():
    if request.method == 'POST':
        data = app_main(request)
        print('data', data)
        return jsonify(data)
    else:
        return jsonify({'message': 'GET method is not supported on this endpoint'}), 405


@main.route('/api/save_image', methods=['POST'])
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


@main.route('/api/profile')
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



@main.route('/api/delete_history/<int:history_id>', methods=['DELETE'])
@login_required
def delete_history(history_id):
    record = History.query.get_or_404(history_id)
    if record.user_id != current_user.id:
        abort(403)  # Forbidden access
    db.session.delete(record)
    db.session.commit()
    return jsonify({'message': 'Record deleted successfully'}), 200


# Create OAuth Object
# oauth_object = spotipy.SpotifyOAuth(spotify_client_id, spotify_client_secret, spotify_redirect_uri, scope=spotify_scope)
# # Create token
# token_dict = oauth_object.get_access_token()
# token = token_dict['access_token']
# Create Spotify Object
spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())



def search_and_play_song(search_keyword):
    # Search for the Song.
    search_results = spotify.search(search_keyword, 1, 0, "track")
    # Get required data from JSON response.
    track = search_results['tracks']
    tracks_items = track['items']
    if tracks_items:
        return tracks_items[0]['external_urls']['spotify']
    else:
        return None


@main.route('/api/playsong', methods=['POST'])
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


@main.route('/api/login', methods=['POST'])
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

@main.route('/api/profile_redirect', methods=['GET'])
@login_required
def api_redirect_to_profile():
    return jsonify({'message': 'Redirect to profile', 'profile_url': url_for('/api/profile')}), 200


@main.route('/')
def index():
    return jsonify({'message': 'Welcome to MindPallette API'}), 200


# if __name__ == '__main__':
#     app.run(debug=True)