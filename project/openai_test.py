from openai import OpenAI
from datetime import datetime
import os
import warnings
from flask import jsonify
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
print(openai_key)

client = OpenAI(
    api_key=openai_key)
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
print('extracted emotions:', map_srl_to_emotion(verbs_and_roles))