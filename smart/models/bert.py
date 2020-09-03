import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertModel, BertPreTrainedModel


class PairedBinaryClassification(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size * 2, 2)
    
    def forward(self, pooled_text_output, pooled_label_output):
        pooled_output = torch.cat((pooled_text_output, pooled_label_output), dim=1)
        output = self.dense(pooled_output)

        return output


class BertModelOutput:
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits


class BertForPairedBinaryClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = PairedBinaryClassification(config)
        
        self.init_weights()

    def forward(self, text_ids, text_masks, label_ids, label_masks, labels=None, return_dict=None):
        text_outputs = self.bert(text_ids, attention_mask=text_masks)
        label_outputs = self.bert(label_ids, attention_mask=label_masks)

        pooled_text_output = self.dropout(text_outputs[1])
        pooled_label_output = self.dropout(label_outputs[1])

        logits = self.classifier(pooled_text_output, pooled_label_output)
        loss = None
        
        if labels is not None:
            loss = CrossEntropyLoss()(logits, labels)

        if return_dict is None:
            return loss if labels is not None else logits

        return BertModelOutput(loss, logits)


class BertForMultipleLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, text_ids, text_masks, labels=None, return_dict=None):
        outputs = self.bert(text_ids, attention_mask=text_masks)

        pooled_output = self.dropout(outputs[1])
        logits = self.classifier(pooled_output)
        loss = None

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.double().view(-1, self.num_labels))

        if return_dict is None:
            return loss if labels is not None else logits

        return BertModelOutput(loss, logits)
