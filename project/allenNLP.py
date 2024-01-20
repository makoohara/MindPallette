import spacy
from spacy import displacy
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
from openai import OpenAI
from datetime import datetime
import random
from dotenv import load_dotenv
import os
import warnings
from flask import jsonify
warnings.filterwarnings("ignore")

#load the model
SRL_MODEL_PATH = "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"

def extract_verbs_and_roles(srl_data):
    verbs_and_roles = []
    for verb in srl_data['verbs']:
        verb_text = verb['verb']
        roles = []
        for tag, word in zip(verb['tags'], srl_data['words']):
            if tag.startswith("B-"):
                role = tag.split("-")[1]
                roles.append((role, word))
        verbs_and_roles.append((verb_text, roles))
    return verbs_and_roles


openai_api_key = os.getenv("OPENAPI_KEY")
client = OpenAI(api_key=openai_api_key)

def opneai_processing(verb):
    system_msg = 'You are a helpful assistant who specialize in figurative emotional exploration, especially in translating contexual events to emotional keywords.'
    prompt = f"What emotion is typically associated with the action '{verb}'?"

    try:
        response = client.chat.completions.create(
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
    except Exception as e:
        # Handling errors by sending an error response
        print('OpenAI Dalle Error', str(e))
        return jsonify({'OpenAI Dalle Error': str(e)}), 500

def map_srl_to_emotion(verbs_and_roles):
    emotions=[]
    for verb, roles in verbs_and_roles:
        emotion = opneai_processing(verb)
        if emotion:
            emotions.append(emotion)
    return emotions

predictor = Predictor.from_path(SRL_MODEL_PATH)
predictions = predictor.predict(sentence="Did Uriah honestly think he could beat the game in under three hours?.")
print('srl_tags:', predictions)
verbs_and_roles = extract_verbs_and_roles(predictions)
print('verbs and roles:', verbs_and_roles)
print('extracted emotions:', map_srl_to_emotion(verbs_and_roles))



"""
{'verbs': [{'verb': 'Did', 'description': '[V: Did] Uriah honestly think he could beat the game in under three hours ? .', 'tags': ['B-V', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']}, {'verb': 'think', 'description': 'Did [ARG0: Uriah] [ARGM-ADV: honestly] [V: think] [ARG1: he could beat the game in under three hours] ? .', 'tags': ['O', 'B-ARG0', 'B-ARGM-ADV', 'B-V', 'B-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'O', 'O']}, {'verb': 'could', 'description': 'Did Uriah honestly think he [V: could] beat the game in under three hours ? .', 'tags': ['O', 'O', 'O', 'O', 'O', 'B-V', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']}, {'verb': 'beat', 'description': 'Did Uriah honestly think [ARG0: he] [ARGM-MOD: could] [V: beat] [ARG1: the game] [ARGM-TMP: in under three hours] ? .', 'tags': ['O', 'O', 'O', 'O', 'B-ARG0', 'B-ARGM-MOD', 'B-V', 'B-ARG1', 'I-ARG1', 'B-ARGM-TMP', 'I-ARGM-TMP', 'I-ARGM-TMP', 'I-ARGM-TMP', 'O', 'O']}], 'words': ['Did', 'Uriah', 'honestly', 'think', 'he', 'could', 'beat', 'the', 'game', 'in', 'under', 'three', 'hours', '?', '.']}

1. allenNLP -> subtract specifically adjustives near a noun and verbs 
2. estimate the emotion the person is feeling 
3. also record the change through time. 

"""