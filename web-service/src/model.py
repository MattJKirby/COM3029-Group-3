from tensorflow import keras
from keras.models import load_model, Model
from keras.layers import Input, Dense
from transformers import AutoTokenizer, TFRobertaForSequenceClassification
import numpy as np
import sys
import os


class Model:
    def __init__(self,modelPath):
        if(os.path.isdir(modelPath)):
          self.model = load_model(modelPath)
          self._tokenizer = AutoTokenizer.from_pretrained("roberta-base")
          self._classes = ['sadness', 'confusion', 'joy', 'anger', 'optimism', 'disapproval', 'love', 'curiosity', 'amusement', 'annoyance', 'gratitude', 'approval', 'admiration', 'neutral']
          
    def _tokenize(self, input):
        return self._tokenizer([input], padding = "max_length", truncation = True, max_length = 31)

    def predict(self, input):
        tokenized = self._tokenizer([input], padding = "max_length", truncation = True, max_length = 31)
        raw_prediction = self.model.predict([np.array(tokenized["input_ids"]), np.array(tokenized["attention_mask"])], verbose = 0)
        return self._classes[np.argmax(raw_prediction)]