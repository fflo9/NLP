import numpy as np
import os
import re
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from datasets import Dataset, DatasetDict
import io
import pandas as pd
import random
import torch

def read_file(path):

     data = []
     current_words = []
     current_tags = []

     for line in open(path, encoding='utf-8'):
          line = line.strip()

          if line:
               if line[0] == '#':
                    continue
               tok = line.split('\t')

               if len(tok) >= 3:
                current_words.append(tok[1])
                current_tags.append(tok[2])

               elif len(tok) == 2:
                    current_words.append(tok[0])
                    current_tags.append(tok[1])

          else:
               if current_words:
                    data.append((current_words, current_tags))

               current_words = []
               current_tags = []

     if current_tags != []:
          data.append((current_words, current_tags))

     return data

def label_config():
    id2label = {
        0: 'O',
        1: 'B-PERSON',
        2: 'I-PERSON',
        3: 'B-LOCATION',
        4: 'I-LOCATION',
        5: 'B-ORGANIZATION',
        6: 'I-ORGANIZATION',
        7: 'B-SYMPTOM',
        8: 'I-SYMPTOM',
        9: 'B-HISTORY',
        10: 'I-HISTORY',
        11: 'B-ACTION',
        12: 'I-ACTION',
        13: 'B-OTHER',
        14: 'I-OTHER'
    }

    label2id = {
        'O': 0,
        'B-PERSON': 1,
        'I-PERSON': 2,
        'B-LOCATION': 3,
        'I-LOCATION': 4,
        'B-ORGANIZATION': 5,
        'I-ORGANIZATION': 6,
        'B-SYMPTOM': 7,
        'I-SYMPTOM': 8,
        'B-HISTORY': 9,
        'I-HISTORY': 10,
        'B-ACTION': 11,
        'I-ACTION': 12,
        'B-OTHER': 13,
        'I-OTHER': 14
    }

    label_list = list(label2id.keys())

    return id2label, label2id, label_list

#reproducibility
def set_seed(seed=52):
     random.seed(seed)
     np.random.seed(seed)
     torch.manual_seed(seed)

     if torch.cuda.is_available():
          torch.cuda.manual_seed_all(seed)

     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False

def get_report_df(report):
    # Check if it's a dictionary
    if isinstance(report, dict):
        return pd.DataFrame.from_dict(report, orient='index')
    # If it's a string, parse it
    else:
        # Use regex separator to catch the multiple spaces between columns
        return pd.read_csv(io.StringIO(report), sep=r'\s{2,}', engine='python')
    

def validate_file(path):
     if not os.path.exists(path):
          raise FileNotFoundError(f'File not found: {path}')
     
     if os.path.getsize(path) == 0:
          raise ValueError(f'File is empty: {path}')
     
def validate_bio(data):
    errors = []

    for sent_idx, (words, tags) in enumerate(data):
        prev_tag = "O"

        for tok_idx, tag in enumerate(tags):
            if tag == "O":
                prev_tag = tag
                continue

            prefix, entity = tag.split("-", 1)

            #invalid I-tag at sentence start
            if prefix == "I":

                if prev_tag == "O":
                    errors.append(
                        (sent_idx, tok_idx, words[tok_idx], tag,
                         "I-tag after O")
                    )

                elif prev_tag != "O":
                    prev_prefix, prev_entity = prev_tag.split("-", 1)

                    if prev_entity != entity:
                        errors.append(
                            (sent_idx, tok_idx, words[tok_idx], tag,
                             "I-tag following different entity")
                        )

            prev_tag = tag

    return errors

def fix_bio(tags):
    fixed = []
    prev = "O"

    for tag in tags:
        if tag == "O":
            fixed.append(tag)
            prev = tag
            continue

        prefix, entity = tag.split("-", 1)

        if prefix == "I":
            if prev == "O":
                tag = f"B-{entity}"

            else:
                prev_prefix, prev_entity = prev.split("-", 1)
                if prev_entity != entity:
                    tag = f"B-{entity}"

        fixed.append(tag)
        prev = tag

    return fixed

def dictionize(train_data, dev_data, test_data, label2id):
    train_dict = {
        'tokens': [],
        'ner_tags': []
    }
    #loop through the train data
    for words, tags in train_data:
        #add the raw str words to the tokens
        train_dict['tokens'].append(words)

        #convert the string tags to ID numbers using label2id map
        tag_ids = [label2id[tag] for tag in tags]
        train_dict['ner_tags'].append(tag_ids)

    #dev data
    dev_dict = {
        'tokens': [],
        'ner_tags': []
    }
    for words, tags in dev_data:
        dev_dict['tokens'].append(words)
        tag_ids = [label2id[tag] for tag in tags]
        dev_dict['ner_tags'].append(tag_ids)

    #test data
    test_dict = {
        'tokens': [],
        'ner_tags': []
    }
    for words, tags in test_data:
        test_dict['tokens'].append(words)
        test_dict['ner_tags'].append([label2id[tag] for tag in tags])

    #convert all 3 dictionaries to a hf dataset then datasetdict
    trainDataset = Dataset.from_dict(train_dict)
    devDataset = Dataset.from_dict(dev_dict)
    testDataset = Dataset.from_dict(test_dict)
    datasetDict = DatasetDict({'train': trainDataset, 'dev': devDataset, 'test': testDataset})

    return datasetDict

def tokenize_and_align_labels(tokenizer, examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=512,
        is_split_into_words=True,
        padding=False,         #let data_collator handle padding
    )
    labels = []
    for i, label in enumerate(examples[f'ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs

def get_predictions(trainer, data): 
    output = trainer.predict(data)
    labels = output.label_ids
    
    #check if predictions are a tuple (common with CRF models)
    if isinstance(output.predictions, tuple):
        preds = output.predictions[0]
    else:
        preds = np.argmax(output.predictions, axis=2)
        
    return preds, labels

def class_report(predictions, labels, label_list):
    #remove -100 (padding) and convert id2label
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return classification_report(true_labels, true_predictions, output_dict=True)

def compute_metrics_model(p, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "f1": f1_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
    }