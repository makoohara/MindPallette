from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, abort
from . import db
from openai import OpenAI
from .models import History
from flask_login import current_user, login_required
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from fastcoref import FCoref
from itertools import islice
from dotenv import load_dotenv
import random
import math
# import nltk
import re
import numpy as np
import os
import spotipy
# from spotipy.oauth2 import SpotifyOAuth
# from allennlp.predictors.predictor import Predictor
# import allennlp_models.tagging


# nltk.download('vader_lexicon')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('stopwords')

# from nltk.tag import pos_tag
# from nltk.chunk import ne_chunk
# from nltk.corpus import stopwords
load_dotenv()

main = Blueprint('main', __name__)
# Define OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
# User credentials and Spotify setup
spotify_client_id = os.getenv("SPOTIFY_CLIENT_ID")
spotify_client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
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
        # srl_result = self.nlp_util.srl(tokenized_sentences)
        # print('srl_result:', srl_result)
        icons = {'main_icons': main_icons, "sub_icons": sub_icons}
        print('icons:', icons)
        data = {
            1: {**icons, **{'image_attributes': self.select_image_attributes(scaled_score)}},
            2: {'keywords': icons, 'overall sentiment': overall_sentiment},
            'normalized overall std': sentiment_stats['normalized_overall_std'],
            3: {'text': entry}
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
        self.model_path = {'coref': "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"}


    def icon_extraction(self, entry, model_path=None, device='cpu'):
        # Initialize the FCoref model
        model = FCoref(device=device)
    
        # Predict coreferences
        preds = model.predict(texts=[entry])
        
        # Extract clusters as strings
        clusters = preds[0].get_clusters()
        print('clusters:', clusters)
        icons = {}
        for cluster in clusters:
            freq = len(cluster)
            word = cluster[0]
            if word in self.diary_stop_words:
                continue
            icons[word] = freq
        
        # Sort mentions by frequency in descending order
        sorted_icons_desc = {k: v for k, v in sorted(icons.items(), key=lambda item: item[1], reverse=True)}

        # Selecting main and sub icons based on frequencies
        main_keyword_index = random.randint(1, 4)
        sub_keyword_index = main_keyword_index + random.randint(1, 5)
        main_icons = dict(islice(sorted_icons_desc.items(), 0, main_keyword_index))
        sub_icons = dict(islice(sorted_icons_desc.items(), main_keyword_index, sub_keyword_index))
        print('sorted icons:', sorted_icons_desc)
        print('Main Icons:', main_icons)
        print('Sub Icons:', sub_icons)
        
        return main_icons, sub_icons

    def tokenize(self, entry):
        sentences = sent_tokenize(entry)
        return sentences


    # def srl(self, tokenized_sentences):
    #     """
    #     Extract verbs from a sentence using SRL, then query ChatGPT for associated emotions.
    #     """
    #     srl_model_path = self.model_path['srl']
    #     predictor = Predictor.from_path(srl_model_path)
    #     result = []
    #     for sentence in tokenized_sentences:
    #         srl_result = predictor.predict(sentence=sentence)
    #         result.append(srl_result)
    #     return result

    # def annotate_srl(self, srl_result):
    #     structured_data = {}
    #     for sentence in srl_result:
    #         verbs = sentence['verbs']
    #         for verb_dict in verbs:
    #             roles = verb_dict['description']
    #             components = re.findall(r'\[(.*?)\]', roles)
    #             for component in components:
    #                 # Split each component into its label and text
    #                 label, text = component.split(': ', 1)
    #                 if label in structured_data:
    #                     structured_data[label] += [text]
    #                 else: 
    #                     structured_data[label] = [text]
    #     return structured_data


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
        system_msg = 'You are a prompt engineer for Dalle. You improvise and adjust prompts parameter based on information provided of diary text. Only return parameters that can directly paste into Dall-E to generate an image. The goal of the prompt is to figuratively express the emotion in the diary. The more detailed and nuanced the prompt, the better.'
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
        elif pipeline == 3:
            text = processed_data['text']
            prompt = f"Return parameters description and its weight to describe the emotional experience as an image for Dalle. The parameter structure should look like this: {parameters}. \n\
                        Your task is to change the item values of each parameter to match this diary text. {text} "
            try:
                print('Dalle prompt', pipeline, ":", prompt)
                return self.query(system_msg, prompt)
            except Exception as e:
                # Handling errors by sending an error response
                print('OpenAI GPT-3.5 Error', str(e))
                return jsonify({'OpenAI GPT-3.5 Error': str(e)}), 500

        
    def generate_image_url(self, prompt):
        dalle_prompt = ' '.join(prompt)
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
        for i in [1, 3]:
            pipeline = i
            print('generating Dall-E prompt with pipeline', pipeline)
            data = processed_data[i]
            print('pipeline', i, data)
            
            prompt = openai_util.generate_parameters(data, pipeline)

            if i == 3 or i == 4:
                prompt = processor.output_cleaning(prompt)

            print('prompt', i, prompt, type(prompt))
            result = openai_util.generate_image_url(prompt)

            print('result', i, result)
            
            results.append(str(result))

        song = openai_util.recommend_song(entry)
        print("song object", song, type(song))
        print("img_urls", results)
        return jsonify({'song': song, 'img_urls': results})

    except Exception as e:
        # Handling errors by sending an error response
        return ({'error': str(e)}, 500)


@main.route('/home', methods=['POST', 'GET'])
@login_required
def home():
    if request.method == 'POST':
        # Check the number of saved images
        saved_images_count = History.query.filter_by(user_id=current_user.id).count()

        # Prompt user to delete history if there are 6 or more saved images
        if saved_images_count >= 6:
            data = {'error': 'You have reached the maximum limit of saved images. Please delete one or more history entries.'}
            return jsonify(data)
        data = app_main(request)
        # if status_code != 200:
        #     return jsonify(data), status_code
        print('data', data, 'type', type(data))
        return data
    else:
        return render_template('index.html', data=None, user=current_user)

@main.route('/save_image', methods=['POST'])
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

    return render_template('profile.html', data=None, user=current_user)


@main.route('/profile')
@login_required
def profile():
    user_history = History.query.filter_by(user_id=current_user.id).order_by(History.date_time.desc()).all()
    return render_template('profile.html', user=current_user, history=user_history)


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


@main.route('/login')
def login():
    return redirect(url_for('main.home'))


@main.route('/')
@login_required
def redirect_to_profile():
    return redirect(url_for('auth.login'))