from openai import OpenAI
from datetime import datetime
import os
import warnings
from flask import jsonify
from dotenv import load_dotenv
import re

warnings.filterwarnings("ignore")
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
print(openai_key)

client = OpenAI(
    api_key=openai_key)


def pipeline3(text):
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
    prompt = f"Return parameters for a Dalle prompt based on this diary {text}. The example parameter is here: {parameters}."
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
verbs_and_roles = [['breathe', 'v']]
# print('extracted emotions:', map_srl_to_emotion(verbs_and_roles))

def generate_image_url(prompt):
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



entry = 'Today was one of those days where Londons grey skies felt more comforting than gloomy. Wrapped in my favorite scarf, the chill in the air somehow matched my mood - reflective, a bit somber, yet hopeful. University life in this sprawling city continues to be an emotional rollercoaster. The blend of excitement and overwhelming moments hasnt faded since the day I arrived.  I spent the morning in the British Library, nestled among books and the silent determination of fellow students. Theres something about that place that makes my worries seem quieter, the weight of deadlines a bit lighter. As I walked back to my flat, the aroma of fresh rain on pavement filled the air, a scent thats become a strange companion in my solitary moments.  Lectures today felt particularly engaging, diving into topics that challenge my perspectives and push my boundaries. Yet, amidst the intellectual stimulation, theres this undercurrent of solitude that I cant seem to shake off. Its odd, being surrounded by a sea of faces, yet feeling a disconnect. I miss the effortless conversations and laughter with friends back home, the familiarity of shared history.  Evening brought a spontaneous adventure - a solo exploration of a little bookshop I stumbled upon. Hidden treasures nestled in its shelves offered a brief escape, a reminder of the simple joys I often overlook. The city, with its endless buzz and hidden quiet corners, never ceases to surprise me.  As night envelops London, the skyline a silhouette of dreams against the twilight, I find solace in writing down these thoughts. Theres a peculiar beauty in navigating this chapter of my life, a tapestry of growth, learning, and self-discovery. Despite the occasional bouts of loneliness, theres a part of me thats grateful for this journey, for the person Im becoming amidst the chaos of city life.  Tomorrow promises another page of this London adventure, another opportunity to embrace the unknown. Until then, Ill hold onto the small victories, the fleeting moments of connection, and the hope that, in time, Ill find my tribe in this vast metropolis.  Goodnight, London.'
parameters = pipeline3(entry)
print('process3: ', parameters)
print('image_ulr_3: ', generate_image_url(parameters))