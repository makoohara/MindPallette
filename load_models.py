from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging


#load the model
SRL_MODEL_PATH = "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz"

predictor = Predictor.from_path(SRL_MODEL_PATH)