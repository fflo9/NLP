import numpy as np
from seqeval.metrics import classification_report
from transformers import Trainer
from transformers import TrainingArguments
from transformers import DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
import torch

# Read the files
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

#aligning the labels
def tokenize_and_align_labels(tokenizer, examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)

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

#trainer with weighted classes
class WeightedTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    labels = inputs.get('labels')

    #forward pass
    outputs = model(**inputs)
    logits = outputs.get('logits')

    weights = torch.tensor([0.5, 2.0, 2.0, 1.2, 1.2, 2.5, 2.5]).to(logits.device)

    loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
    loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

    return (loss, outputs) if return_outputs else loss

#get the classification report
def compute_metrics(predictions, labels, label_list):

    #remove -100 (padding) and convert id2label
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return classification_report(true_labels, true_predictions)

def predictions(trainer, data, label_list):
    #get predictions
    output = trainer.predict(data)
    predictions = np.argmax(output.predictions, axis=2) #the label w/ the highest score
    
    #ground truth
    labels = output.label_ids

    return predictions, labels

#save the results
def save_results(file_path, data, pred_array, tokenized_data, label_list):
    with open(f'{file_path}.iob2', 'w', encoding='utf-8') as file:
        for i, sentence_data in enumerate(data):
            original_words = sentence_data[0]
            word_ids = tokenized_data[i]['input_ids']
            preds = pred_array[i]

            # Align subwords
            word_predictions = []
            previous_word_idx = None
            for j, word_idx in enumerate(word_ids):
                if word_idx is None: continue
                if word_idx != previous_word_idx:
                    word_predictions.append(label_list[preds[j]])
                previous_word_idx = word_idx

            # 3. Write using \t (Tabs)
            for idx, (word, pred_tag) in enumerate(zip(original_words, word_predictions)):
                # Force tab separation
                file.write(f'{idx}\t{word}\t{pred_tag}\n')

            # Sentence break
            file.write('\n')

    print('File saved successfully')
