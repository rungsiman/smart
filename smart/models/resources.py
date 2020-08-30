import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertPreTrainedModel


class OntologyClassification(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size * 2, 2)
    
    def forward(self, pooled_text_output, pooled_label_output):
        pooled_output = torch.cat((pooled_text_output, pooled_label_output), dim=1)
        output = self.dense(pooled_output)

        return output


class BertForOntologyClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = OntologyClassification(config)
        
        self.init_weights()

    def forward(self, text_ids, text_masks, label_ids, label_masks, tags=None):
        text_outputs = self.bert(text_ids, attention_mask=text_masks)
        label_outputs = self.bert(label_ids, attention_mask=label_masks)

        pooled_text_output = self.dropout(text_outputs[1])
        pooled_label_output = self.dropout(label_outputs[1])

        logits = self.classifier(pooled_text_output, pooled_label_output)
        
        if tags is not None:
            loss = CrossEntropyLoss()(logits, tags)
            return loss, logits
        
        else:
            return logits
