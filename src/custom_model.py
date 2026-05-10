from torchcrf import CRF
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import BertModel
import torch.nn as nn
from transformers import Trainer

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get('labels')
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
        
    def _save(self, output_dir, state_dict=None):
        if state_dict is None:
            state_dict = self.model.state_dict()
        # make all tensors contiguous before saving
        state_dict = {k: v.contiguous() for k, v in state_dict.items()}
        super()._save(output_dir, state_dict=state_dict)
        
class BertCRFForNER(nn.Module):
    def __init__(self, bert_model_name, num_labels, id2label, label2id):
        super().__init__()
        self.num_labels = num_labels
        self.config = type('config', (), {'num_labels': num_labels, 'id2label': id2label, 'label2id': label2id})()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)
    
        loss = None
        if labels is not None:
            crf_labels = labels.clone()
            crf_mask = labels != -100
            crf_labels[~crf_mask] = 0
            
            # CRF requires first timestep to always be unmasked
            crf_mask[:, 0] = True
            crf_labels[:, 0] = 0  # assign O label to [CLS] token
            
            loss = -self.crf(emissions, crf_labels, mask=crf_mask, reduction='mean')
    
        return TokenClassifierOutput(loss=loss, logits=emissions)