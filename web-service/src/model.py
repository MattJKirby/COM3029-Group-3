from tensorflow import keras
from transformers import AutoTokenizer
import numpy as np

class Model:
    def __init__(self,modelPath):
        self.model = keras.models.load_model(modelPath)
        self._tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self._classes = ['sadness', 'confusion', 'joy', 'anger', 'optimism', 'disapproval', 'love', 'curiosity', 'amusement', 'annoyance', 'gratitude', 'approval', 'admiration', 'neutral']
        
    def _tokenize(self, input):
        return self._tokenizer([input], padding = "max_length", truncation = True, max_length = 31)

    def predict(self, input):
        tokenized = self._tokenize(input)
        prediction = self.model.predict([np.array(tokenized["attention_mask"]), np.array(tokenized["input_ids"])])
        return self._classes[np.argmax(prediction)]