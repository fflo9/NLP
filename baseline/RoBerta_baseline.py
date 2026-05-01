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

if __name__ == "__main__":
    #load train and dev data
    train_data = read_file(r'baseline/en_ewt-ud-train.iob2')
    dev_data = read_file(r'baseline/en_ewt-ud-dev.iob2')
    test_data = read_file(r'baseline/en_ewt-ud-test.iob2')

    id2label = {
        0: "O",
        1: "B-LOC",
        2: "I-LOC",
        3: "B-PER",
        4: "I-PER",
        5: "B-ORG",
        6: "I-ORG",
    }
    label2id = {
        "O": 0,
        "B-LOC": 1,
        "I-LOC": 2,
        "B-PER": 3,
        "I-PER": 4,
        "B-ORG": 5,
        "I-ORG": 6,
    }

    #set the tokenizer 
    tokenizer = RobertaTokenizerFast.from_pretrained('FacebookAI/roberta-base', add_prefix_space = True)

    #automatic padding, and converts data to tensors
    data_collator = DataCollatorForTokenClassification(tokenizer)

    #initialize model
    model = RobertaForTokenClassification.from_pretrained(
    'FacebookAI/roberta-base', num_labels=7, id2label=id2label, label2id=label2id
    )

    #initialize training arguements
    training_args = TrainingArguments(
        output_dir='./roberta-ner-results',
        remove_unused_columns=False,
        eval_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        save_strategy='epoch',
        gradient_accumulation_steps=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        weight_decay=0.01,
    )

    #tokenize all splits
    datasetDict = dictionize(train_data, dev_data, test_data, label2id)
    tokenizedData = datasetDict.map(
        lambda examples: tokenize_and_align_labels(tokenizer, examples), 
        batched=True
    )

    #remove raw columns for all splits
    train_final = tokenizedData['train'].remove_columns(['tokens', 'ner_tags'])
    dev_final = tokenizedData['dev'].remove_columns(['tokens', 'ner_tags'])
    test_final = tokenizedData['test'].remove_columns(['tokens', 'ner_tags'])

    #set trainer and start training
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_final,
        eval_dataset=dev_final,
        data_collator=data_collator,
    )

    trainer.train()

    #the IOB tags
    label_list = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG']

    #predictions and classification report
    pred_array, labels = predictions(trainer, test_final, label_list)
    class_report = compute_metrics(pred_array, labels, label_list)
    print(class_report)

    #save results
    save_results('test_predictions', test_data, pred_array, tokenizedData['test'], label_list)
