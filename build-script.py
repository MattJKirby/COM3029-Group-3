import tensorflow as tf

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import numpy as np

# nltk
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Spelling imports
from language_tool_python import LanguageTool

# Huggingface
from transformers import TFRobertaForSequenceClassification, AutoTokenizer
from datasets import load_dataset, DatasetDict, logging


logging.set_verbosity_error()
logging.disable_progress_bar()

# Classname Choice -------------------------------------------------------------

raw_dataset = load_dataset("go_emotions")

all_class_names = ["admiration", "amusement", "anger", "annoyance", "approval",
                   "caring", "confusion", "curiosity", "desire", "disappointment",
                   "disapproval", "disgust", "embarrassment", "excitement", "fear",
                   "gratitude", "grief", "joy", "love", "nervousness", "optimism",
                   "pride", "realization", "relief", "remorse", "sadness", "surprise",
                   "neutral"]

# Get top 14 most frequently occuring keys in dataset and fetch their indices
class_counts = {class_name: 0 for class_name in all_class_names}

for item in raw_dataset['train']:
  for class_name in item['labels']:
    class_counts[all_class_names[class_name]] += 1

feature_distribution = dict(sorted(class_counts.items(), key=lambda item: item[1]))

class_names = list(feature_distribution.keys())[-14:]
class_name_idxs = [all_class_names.index(x) for x in class_names]
print(list(zip(class_names, class_name_idxs)))

# Dataset Filtering -------------------------------------------------------------

# If it has at least one label that is the selected subset of classes it's valid
def is_valid(data_item):
  return not (len(data_item["labels"]) == 1 and data_item["labels"][0] not in class_name_idxs)

# Remove classes that don't have a label in our 14 selected classes 
def remove_invalid_classes(data_item):
  data_item["labels"] = [label for label in data_item["labels"] if label in class_name_idxs] 

  for label in data_item["labels"]:
    assert label in class_name_idxs

  data_item["labels"] = [class_name_idxs.index(label) for label in data_item["labels"]][0:1] # "Rename" old labels

  return data_item

def one_hot_labels(data_item):
  data_item["labels"] = np.sum(to_categorical(data_item["labels"], len(class_name_idxs)), axis = 0)
  return data_item

# Apply dataset processing
dataset = raw_dataset.filter(lambda x: is_valid(x)).map(remove_invalid_classes)

# One-hot the labels
dataset_base = dataset.map(one_hot_labels)

# Lemmatizing With POS Tagging -------------------------------------------------------------

def lemm_with_pos_tagging(example):
  tokens = word_tokenize(example['text'])
  tagged = pos_tag(tokens)
  lemmatizer = WordNetLemmatizer()
  pos_tags = {'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV, 'J': wordnet.ADJ}
  words = []
  for word, tag in tagged:
    if tag[0] in pos_tags:
      words.append(lemmatizer.lemmatize(word, pos=pos_tags[tag[0]]))
    else:
      words.append(lemmatizer.lemmatize(word))
  example['text'] = ' '.join(words)
  return example

dataset_base['train'] = dataset_base['train'].map(lemm_with_pos_tagging)

# Grammar Correction -------------------------------------------------------------

lang_tool = LanguageTool('en-US')

def correct_grammar(example):
  sentence = example['text']
  errors = lang_tool.check(sentence)
  
  if len(errors) > 0:
    for error in reversed(errors):
      if len(error.replacements) > 0:
        corrected = sentence[:error.offset] + error.replacements[0] + sentence[error.offset + error.errorLength:]
        sentence = corrected
        example['text'] = ''.join(sentence)
  return example

logging.enable_progress_bar()
dataset_base = dataset_base.map(correct_grammar)
logging.disable_progress_bar()

idx = np.random.randint(0, 2000)

print(dataset_base['train'][idx]["text"])
print(class_names[np.argmax(dataset_base['train'][idx]["labels"])])

dataset_base = DatasetDict(dataset_base)

# Dataset Tokenization -------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

seq_lens = [len(tokenizer(x)["input_ids"]) for x in dataset_base["train"]["text"]]
final_seq_len = int(np.ceil(np.mean(seq_lens) + np.std(seq_lens)))

def tokenize_dataset(data):
    # Keys of the returned dictionary will be added to the dataset as columns
    tokenizer_out = tokenizer(data["text"], padding = "max_length", truncation = True, max_length = final_seq_len) # Sets length of tokenized string to mean token sequence length
    for key in tokenizer_out:
      data[key] = tokenizer_out[key]
    return data

dataset_tokenized = dataset_base.map(tokenize_dataset)

# Dataset to tf.data -------------------------------------------------------------

batch_size = 128

train_dataset = dataset_tokenized["train"].to_tf_dataset(
  columns = ["input_ids", "attention_mask"],
  label_cols = ["labels"],
  batch_size = batch_size,
  shuffle = True,
)

test_dataset = dataset_tokenized["test"].to_tf_dataset(
  columns = ["input_ids", "attention_mask"],
  label_cols = ["labels"],
  batch_size = batch_size,
  shuffle = True,
)

# Training -------------------------------------------------------------

def define_model():   
  input_ids = Input(shape = (final_seq_len,), dtype = "int32", name = "input_ids")
  attention_masks = Input(shape = (final_seq_len,), dtype = "int32", name = "attention_mask")

  inputs = {"input_ids": input_ids, "attention_mask": attention_masks}

  model = TFRobertaForSequenceClassification.from_pretrained("roberta-base", num_labels = 14)(inputs).logits
  model = Dense(14, activation = "softmax")(model)

  model = Model(inputs = [input_ids, attention_masks], outputs = model)

  optimizer = tf.keras.optimizers.Adagrad(learning_rate = 1e-3)

  model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"])

  return model

model = define_model()

early_stopping = EarlyStopping(monitor = "val_loss", patience = 7, restore_best_weights = True)

model.fit(train_dataset, validation_data = test_dataset, epochs = 25, callbacks = [early_stopping])

model.save("models/model")