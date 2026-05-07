from RoBerta_baseline import read_file, dictionize, tokenize_and_align_labels, WeightedTrainer, compute_metrics, predictions, save_results
import numpy as np
from seqeval.metrics import classification_report
from transformers import Trainer
from transformers import TrainingArguments
from transformers import DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
import torch
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

#load train and dev data
train_data = read_file(r'train.iob2')
dev_data = read_file(r'validation.iob2')
test_data = read_file(r'test.iob2')

id2label = {
    0: 'O',
    1: 'B-PERSON',
    2: 'I-PERSON',
    3: 'B-LOCATION',
    4: 'I-LOCATION',
    5: 'B-ORGANIZATION',
    6: 'I-ORGANIZATION',
    7: 'B-SYMPTHOM',
    8: 'I-SYMPTHOM',
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
    'B-SYMPTHOM': 7,
    'I-SYMPTHOM': 8,
    'B-HISTORY': 9,
    'I-HISTORY': 10,
    'B-ACTION': 11,
    'I-ACTION': 12,
    'B-OTHER': 13,
    'I-OTHER': 14
}

#the IOB tags
label_list = ['O', 'B-PERSON', 'I-PERSON', 'B-LOCATION', 'I-LOCATION', 'B-ORGANIZATION', 'I-ORGANIZATION', 'B-SYMPTHOM', 'I-SYMPTHOM', 'B-HISTORY', 'I-HISTORY', 'B-ACTION', 'I-ACTION', 'B-OTHER', 'I-OTHER']

#set the tokenizer 
tokenizer = RobertaTokenizerFast.from_pretrained('FacebookAI/roberta-base', add_prefix_space = True)

#automatic padding, and converts data to tensors
data_collator = DataCollatorForTokenClassification(tokenizer)

#initialize model
model = RobertaForTokenClassification.from_pretrained(
'FacebookAI/roberta-base', num_labels=len(label_list), id2label=id2label, label2id=label2id
)

#initialize training arguements
training_args = TrainingArguments(
     seed=42,
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
     batched=True,
)

#remove raw columns for all splits
cols_to_remove = ['tokens', 'ner_tags'] #list of cols to get rid of
tokenizedData = tokenizedData.remove_columns(cols_to_remove)

train_final = tokenizedData['train']
dev_final   = tokenizedData['dev']
test_final  = tokenizedData['test']

#set trainer and start training
trainer = Trainer(
     model=model,
     args=training_args,
     train_dataset=train_final,
     eval_dataset=dev_final,
     data_collator=data_collator,
)

trainer.train()

#dev predictions and classification report
pred_array, labels = predictions(trainer, dev_final, label_list)
class_report = compute_metrics(pred_array, labels, label_list)
print(f'medmentions dev results', class_report)

#test results and save results
#pred_array, labels = predictions(trainer, test_final, label_list)
#class_report = compute_metrics(pred_array, labels, label_list)
#print('medmentions test results': class_report)
#save_results('test_predictions', test_data, pred_array, tokenizedData['test'], label_list)

#load converstaional test
# conv_data = read_file('conversational_test.iob2')
# conv_datasetDict = dictionize([],[], conv_data, label2id)
# conv_tokenized = conv_datasetDict.map(
#      lambda examples: tokenize_and_align_labels(tokenizer, examples),
#      batched=True
# )

# conv_final = conv_tokenized['test'].remove_columns(['tokens', 'ner_tags'])

# conv_pred_array, conv_labels = predictions(trainer, conv_final, label_list)
# conv_class_report = compute_metrics(conv_pred_array, conv_labels, label_list)

# print(f'converstaional data results', conv_class_report)



