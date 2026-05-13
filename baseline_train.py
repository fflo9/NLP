import torch
import random
import os
import io
import numpy as np
import pandas as pd
from datasets import Dataset
from src import (
     validate_file,
     read_file, 
     dictionize, 
     tokenize_and_align_labels, 
     compute_metrics_model, 
     predictions, 
     class_report,
     validate_bio,
     fix_bio
)

from transformers import (
     Trainer,
     TrainingArguments,
     DataCollatorForTokenClassification, 
     RobertaTokenizerFast, 
     RobertaForTokenClassification
)

#reproducibility
def set_seed(seed=52):
     random.seed(seed)
     np.random.seed(seed)
     torch.manual_seed(seed)

     if torch.cuda.is_available():
          torch.cuda.manual_seed_all(seed)

     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False

#label config
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

#main training pipeline
def main():

     try:
          set_seed(52)
          if not os.path.exists('results'):
               os.makedirs('results')

          #data paths
          train_path = r'data_iob2/train.iob2'
          dev_path = r'data_iob2/validation.iob2'
          test_path = r'data_iob2/test.iob2'
          mts_data_path = r'data_iob2/mts_data.iob2'
          paths = [train_path, dev_path, test_path, mts_data_path]

          #validate files
          for path in paths:
               validate_file(path)

          #load data
          train_org = read_file(train_path)
          dev_org = read_file(dev_path)
          test_org = read_file(test_path)
          mts_data = read_file(mts_data_path)

          #check for BIO tagging errors
          errors = validate_bio(train_org)
          print("Number of BIO errors:", len(errors))

          #fix any error
          train_data = [(words, fix_bio(tags))for words, tags in train_org]
          dev_data = [(words, fix_bio(tags))for words, tags in dev_org]
          test_data= [(words, fix_bio(tags))for words, tags in test_org]

          print(f'Train samples: {len(train_data)}')
          print(f'Dev samples: {len(dev_data)}')
          print(f'Test samples: {len(test_data)}')

          #set the tokenizer 
          print('Setting tokenizer!')
          tokenizer = RobertaTokenizerFast.from_pretrained('FacebookAI/roberta-base', add_prefix_space = True)

          #automatic padding, and converts data to tensors
          data_collator = DataCollatorForTokenClassification(tokenizer)

          #initialize model
          print('Initializing model!')
          model = RobertaForTokenClassification.from_pretrained(
          'FacebookAI/roberta-base', num_labels=len(label_list), id2label=id2label, label2id=label2id
          )

          #data processing
          print('Processing data!')
          datasetDict = dictionize(train_data, dev_data, test_data, label2id)
          
          #tokenize all splits
          tokenizedData = datasetDict.map(
               lambda examples: tokenize_and_align_labels(tokenizer, examples), 
               batched=True,
          )

          #remove raw columns for all splits
          cols_to_remove = ['tokens', 'ner_tags'] #list of cols to get rid of
          tokenizedData = tokenizedData.remove_columns(cols_to_remove)

          #final data to be fed into the model
          train_final = tokenizedData['train']
          dev_final   = tokenizedData['dev']
          test_final  = tokenizedData['test']

          #initialize training arguements
          print('Initializing training arguments!')
          training_args = TrainingArguments(
               seed=42,
               output_dir='results/roberta-ner-results',
               warmup_steps=100,
               remove_unused_columns=False,
               eval_strategy='steps',
               eval_steps=250,
               learning_rate= 2e-5,
               per_device_train_batch_size=16,
               per_device_eval_batch_size=16,
               num_train_epochs=10,
               save_strategy='steps',
               save_steps=250,
               load_best_model_at_end=True,
               metric_for_best_model='f1',
               fp16=torch.cuda.is_available()
          )


          #set trainer and start training
          trainer = Trainer(
               model=model,
               args=training_args,
               train_dataset=train_final,
               eval_dataset=dev_final,
               data_collator=data_collator,
               compute_metrics=lambda p: compute_metrics_model(p, label_list)
          )

          print('Training started!')
          trainer.train()

          #evaluation
          print('Evaluating!')
          pred_array, labels = predictions(trainer, dev_final)
          class_report_result = class_report(pred_array, labels, label_list)

          #DEBUG
          def get_report_df(report):
               # Check if it's a dictionary
               if isinstance(report, dict):
                    return pd.DataFrame.from_dict(report, orient='index')
               # If it's a string (like your debug showed), parse it
               else:
                    # Use regex separator to catch the multiple spaces between columns
                    return pd.read_csv(io.StringIO(report), sep=r'\s{2,}', engine='python')
               
          df_dev = get_report_df(class_report_result)
          print(f'MedMentions Dev Results:\n{df_dev}')

          pred_array_test, labels_test = predictions(trainer, test_final)
          class_report_result_test = class_report(pred_array_test, labels_test, label_list)
          df_test = get_report_df(class_report_result_test)
          print(f'MedMentions Test Results:\n{df_test}')

          #save results
          df_test.to_csv('results/MM_test_results_baseline.csv', index=True)

          #load, dictionize and tokenize mts_data for final testing
          print(len(mts_data))
          mts_dict = {
          'tokens': [words for words, tags in mts_data],
          'ner_tags': [[label2id[tag] for tag in tags] for words, tags in mts_data]
          }

          mts_dataset = Dataset.from_dict(mts_dict)

          mts_tokenized = mts_dataset.map(
          lambda examples: tokenize_and_align_labels(tokenizer, examples),
          batched=True
          )

          mts_tokenized = mts_tokenized.remove_columns(['tokens', 'ner_tags']) #mts_final

          print('Evaluating MTS-Dialog Results!')
          pred_array_mts, labels_mts = predictions(trainer, mts_tokenized)
          class_report_result_mts = class_report(pred_array_mts, labels_mts, label_list)
          df_mts = get_report_df(class_report_result_mts)
          
          print(f'MTS-Dialog Test Results:\n{df_mts}')

          #save results
          df_mts.to_csv('results/MTS_results_baseline.csv', index=True)

     except FileNotFoundError as e:
          print(f"[FILE ERROR] {e}")

     except ValueError as e:
          print(f"[VALUE ERROR] {e}")

     except RuntimeError as e:
          print(f"[RUNTIME ERROR] {e}")

     except Exception as e:
          print(f"[UNEXPECTED ERROR] {e}")

if __name__ == "__main__":
    main()