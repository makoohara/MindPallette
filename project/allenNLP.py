# from allennlp.predictors.predictor import Predictor
# import allennlp_models.tagging
# import openai
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import random
from openai import OpenAI
from datetime import datetime
import os
import warnings
from flask import jsonify
from dotenv import load_dotenv
import re
import transformers 
from fastcoref import FCoref
from collections import Counter
from itertools import islice
nltk.download('punkt')


# Load environment variables and OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = openai_api_key

# Load the Semantic Role Labeling model

client = OpenAI(api_key=openai_api_key)



warnings.filterwarnings("ignore")
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=openai_key)


def pipeline3(text, client):
    system_msg = 'You are a prompt engineer for Dalle. You change parameters based on inputs and returns parameter as an python object.'
    parameters = {'main subjects': {'London': 1}, 
                    'secondary subjects': {'orange': 1}, 
                    'aspect ratio': '1:1', 
                    'medium': 'abstract painting', 
                    'camera': {'portrait view': 1}, 
                    'descriptor': {'realism': 1}, 
                    'artist': {'Andy Warhol': 1}, 
                    'lighting': {'black light': 1}, 
                    'color': {'dark': 1}}
    prompt = f"Return parameters for a Dalle prompt based on this diary {text}. The example parameter is here: {parameters}. \n\
    Use this keywords and sentiment descriptive states of the diary as a reference. "
    song_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}],
        temperature=0,
        top_p=1,
        frequency_penalty=0,    
        presence_penalty=0
    )

    parameters = song_response.choices[0].message.content
    pattern = r"\{[^{}]*\}"
    match = re.search(pattern, text)
    if match:
        dict_str = match.group(0)
        # Assuming the extracted string is safe to evaluate
        parameters = eval(dict_str)
        print(parameters)
    return parameters


def map_srl_to_emotion(verbs_and_roles):
    emotions=[]
    for verb, roles in verbs_and_roles:
        emotion = opneai_processing(verb)
        if emotion:
            emotions.append(emotion)
    return emotions
verbs_and_roles = [['breathe', 'v']]
# print('extracted emotions:', map_srl_to_emotion(verbs_and_roles))

def generate_image_url(prompt, client):
    #dalle_prompt = ' '.join(prompt) + ' ,artistic'
    response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
                )
    img_url = response.data[0].url
    return img_url


def icon_extraction_fcoref(entry, model_path=None, device='cpu'):
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

# Example usage
entry = 'Today was one of those days where Londons grey skies felt more comforting than gloomy. Wrapped in my favorite scarf, the chill in the air somehow matched my mood - reflective, a bit somber, yet hopeful. University life in this sprawling city continues to be an emotional rollercoaster. The blend of excitement and overwhelming moments hasnt faded since the day I arrived.  I spent the morning in the British Library, nestled among books and the silent determination of fellow students. Theres something about that place that makes my worries seem quieter, the weight of deadlines a bit lighter. As I walked back to my flat, the aroma of fresh rain on pavement filled the air, a scent thats become a strange companion in my solitary moments.  Lectures today felt particularly engaging, diving into topics that challenge my perspectives and push my boundaries. Yet, amidst the intellectual stimulation, theres this undercurrent of solitude that I cant seem to shake off. Its odd, being surrounded by a sea of faces, yet feeling a disconnect. I miss the effortless conversations and laughter with friends back home, the familiarity of shared history.  Evening brought a spontaneous adventure - a solo exploration of a little bookshop I stumbled upon. Hidden treasures nestled in its shelves offered a brief escape, a reminder of the simple joys I often overlook. The city, with its endless buzz and hidden quiet corners, never ceases to surprise me.  As night envelops London, the skyline a silhouette of dreams against the twilight, I find solace in writing down these thoughts. Theres a peculiar beauty in navigating this chapter of my life, a tapestry of growth, learning, and self-discovery. Despite the occasional bouts of loneliness, theres a part of me thats grateful for this journey, for the person Im becoming amidst the chaos of city life.  Tomorrow promises another page of this London adventure, another opportunity to embrace the unknown. Until then, Ill hold onto the small victories, the fleeting moments of connection, and the hope that, in time, Ill find my tribe in this vast metropolis.  Goodnight, London.'


icon_extraction_fcoref(entry)

def icon_extraction(entry):
    COREF_MODEL_PATH = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
    predictor = Predictor.from_path(COREF_MODEL_PATH)
    predictions = predictor.predict(document=entry)
    print('predictions:', predictions)
    document = predictions['document']
    icons = dict()
    clusters = predictions['clusters']
    for cluster in clusters:
        freq = len(cluster)
        cluster_index = cluster[0]
        word = " ".join(document[cluster_index[0]:cluster_index[1]+1])
        icons[word] = freq
    print('icons:', icons)
    sorted_icons_desc = {word: freq for word, freq in sorted(icons.items(), key=lambda item: item[1], reverse=True)}
    print('sorted_icons_desc:', sorted_icons_desc)
    return sorted_icons_desc


def tokenize(entry):
    sentences = sent_tokenize(entry)
    return sentences


def srl(tokenized_sentences):
    """
    Extract verbs from a sentence using SRL, then query ChatGPT for associated emotions.
    """
    srl_model_path = "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
    predictor = Predictor.from_path(srl_model_path)
    result = []
    for sentence in tokenized_sentences:
        srl_result = predictor.predict(sentence=sentence)
        result.append(srl_result)
    return result


def annotate_srl(srl_result):
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


def sentiment_analysis(tokenized_sentences):
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


def sentiment_stats(sentiment_analysis):
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


def pipeline4(text, annotations, sentiments, client):
    system_msg = 'You are a prompt engineer for Dalle. You adjust parameters based on annotated text inputs and returns parameter as an python object. The goal of the prompt is to visually and figuratively express the emotion in the diary. The more detailed and nuanced the better.'
    parameters = {'main subjects': {'London': 1}, 
                    'secondary subjects': {'orange': 1}, 
                    'aspect ratio': '1:1', 
                    'medium': 'abstract painting', 
                    'camera': {'portrait view': 1}, 
                    'descriptor': {'realism': 1}, 
                    'artist': {'Andy Warhol': 1}, 
                    'lighting': {'black light': 1}, 
                    'color': {'dark': 1}}
    prompt = f"Return parameters for a Dalle prompt based on a diary. The parameter structure should look like this: {parameters}. \n\
    Your task is to change the item values of the parameter dictionary to match this annotated text and sentiment states from the diary. {annotations}, {sentiments} \n\
        Do not contain any word that may violate OpenAI's use case policy. Do not put any artist names."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}],
        temperature=0,
        top_p=1,
        frequency_penalty=0,    
        presence_penalty=0
    )

    parameters = response.choices[0].message.content
    print('parameters output from openai:', parameters)
    pattern = r"\{[^{}]*\}"
    match = re.search(pattern, text)
    if match:
        dict_str = match.group(0)
        # Assuming the extracted string is safe to evaluate
        parameters = eval(dict_str)
        print(parameters)
    return parameters

def extract_coreferences_spacy(document):
    nlp = spacy.load('en_core_web_sm')
    neuralcoref.add_to_pipe(nlp)
    doc = nlp(document)

    for cluster in doc._.coref_clusters:
        # Example: print the cluster
        print(cluster)


# tokenized_sentences = tokenize(entry)
# srl_result = srl(tokenized_sentences)
# annotation = annotate_srl(srl_result)
# sentiments = sentiment_analysis(tokenized_sentences)
# stats = sentiment_stats(sentiments)
# icons = icon_extraction(entry)
# print('icons:', icons)
# print('stats:', stats)
# print('annotation:', annotation)
# print('sentiment analysis:', sentiments)

# parameter_4 = pipeline4(entry, annotation, sentiments, client)
# print('parameter_4:', parameter_4)
# print('image_ulr_4: ', generate_image_url(parameter_4, client))

# compound = sentiments['compound']
# pos = sentiments['positive']
# neg = sentiments['negative']
# plt.plot(compound, marker='o')  # 'o' creates circular markers for each data point
# # Add a title and labels (optional, but recommended for clarity)
# plt.title('Plot of compound sentiment scores for each sentence in the entry')
# plt.xlabel('Index')
# plt.ylabel('sentiment score')
# plt.show()

# # Display the plot

# plt.scatter(neg, pos)
# plt.title('Sample Scatter Plot')
# plt.xlabel('X Values')
# plt.ylabel('Y Values')
# plt.show()

# plt.hist(compound, density=True, label='compound', color='green')  # 'o' creates circular markers for each data point
# plt.hist(pos, density=True, label='positive', color='orange')
# plt.hist(neg, density=True, label='negative', color='blue')
# plt.title('sentiment')
# plt.xlabel('sentiment score')
# plt.ylabel('density')
# plt.show()
