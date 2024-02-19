from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables and OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
#openai.api_key = openai_api_key

# Load the Semantic Role Labeling model
srl_model_path = "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
predictor = Predictor.from_path(srl_model_path)
client = OpenAI(api_key=openai_api_key)
def get_emotion_for_verb(verb):
    """
    Query ChatGPT for the emotion associated with a specific verb.
    """
    system_msg = "You will respond with python object only. "
    prompt = f"List the emotions typically associated with the action '{verb}' in Python list format. If unknown, take a reasonable guess. Avoid unnecessary words and do not make the response a sentence. return in a format that could directly be used in the code."
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
        # response = client.Completion.create(
        #     engine="gpt-3.5-turbo",
        #     prompt=prompt,
        #     max_tokens=50,
        #     temperature=0.5
        # )
        # emotion = response.choices[0].message.content

        #emotion = response.choices[0].text.strip()
        emotion = response.choices[0].message.content
        return emotion
    except Exception as e:
        print(f"Error querying OpenAI for verb '{verb}': {e}")
        return "unknown"

def extract_verbs_and_query_emotions(sentence):
    """
    Extract verbs from a sentence using SRL, then query ChatGPT for associated emotions.
    """
    srl_result = predictor.predict(sentence=sentence)
    verbs = [verb['verb'] for verb in srl_result['verbs']]
    emotions = [get_emotion_for_verb(verb) for verb in verbs]
    return emotions

# Example usage
sentence = "Did Uriah honestly think he could beat the game in under three hours?"
emotions = extract_verbs_and_query_emotions(sentence)
print(f"Extracted emotions: {emotions}")
