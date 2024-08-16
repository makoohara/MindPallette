from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

# Load the model paths
SRL_MODEL_PATH = "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"
COREF_MODEL_PATH = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"

# Load the SRL model
srl_predictor = Predictor.from_path(SRL_MODEL_PATH)
srl_predictions = srl_predictor.predict(sentence="Seoul is a vibrant city.")

print('SRL predictions:', srl_predictions)

# Load the Coreference model
coref_predictor = Predictor.from_path(COREF_MODEL_PATH)
coref_predictions = coref_predictor.predict(document="Seoul is a vibrant city. It is the capital of South Korea.")

print('Coref predictions:', coref_predictions)
