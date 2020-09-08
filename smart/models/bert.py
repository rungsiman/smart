import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, MSELoss
from transformers import BertModel, BertPreTrainedModel

from smart.models.gans import GanForMultipleLabelClassification
from smart.models.gans import GanForPairedLabelClassification
from smart.models.gans import GanForSequenceClassification


class BertModelOutput:
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits


class BertGanModelOutput:
    def __init__(self, d_loss, g_loss, logits):
        self.d_loss = d_loss
        self.g_loss = g_loss
        self.logits = logits


class BertForMultipleLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, text_ids, text_masks, labels=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(text_ids, attention_mask=text_masks)

        pooled_output = self.dropout(outputs[1])
        logits = self.classifier(pooled_output)
        loss = None

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.float().view(-1, self.num_labels))

        if return_dict is None:
            return loss if labels is not None else logits

        return BertModelOutput(loss, logits)


class BertForPairedBinaryClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, 1)

        self.init_weights()

    def forward(self, text_ids, text_masks, label_ids, label_masks, labels=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.bert(text_ids, attention_mask=text_masks)
        label_outputs = self.bert(label_ids, attention_mask=label_masks)

        pooled_text_output = self.dropout(text_outputs[1])
        pooled_label_output = self.dropout(label_outputs[1])
        pooled_output = torch.cat((pooled_text_output, pooled_label_output), dim=1)

        logits = self.classifier(pooled_output)
        loss = None

        if labels is not None:
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

        if return_dict is None:
            return loss if labels is not None else logits

        return BertModelOutput(loss, logits)


class BertGanForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = GanForSequenceClassification(config)

        self.init_weights()

    def forward(self, text_ids, text_masks, labels=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(text_ids, attention_mask=text_masks)
        pooled_output = self.dropout(outputs[1])
        classified = self.classifier(pooled_output, labels)

        if return_dict is None:
            return classified

        return BertGanModelOutput(*classified)


class BertGanForMultipleLabelClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = GanForMultipleLabelClassification(config)

        self.init_weights()

    def forward(self, text_ids, text_masks, labels=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(text_ids, attention_mask=text_masks)
        pooled_output = self.dropout(outputs[1])
        classified = self.classifier(pooled_output, labels)

        if return_dict is None:
            return classified

        return BertGanModelOutput(*classified)


class BertGanForPairedBinaryClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        config.num_labels = 1
        config.hidden_size *= 2

        self.classifier = GanForPairedLabelClassification(config)

    def forward(self, text_ids, text_masks, label_ids, label_masks, labels=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.bert(text_ids, attention_mask=text_masks)
        label_outputs = self.bert(label_ids, attention_mask=label_masks)

        pooled_text_output = self.dropout(text_outputs[1])
        pooled_label_output = self.dropout(label_outputs[1])
        pooled_output = torch.cat((pooled_text_output, pooled_label_output), dim=1)

        classified = self.classifier(pooled_output, labels)

        if return_dict is None:
            return classified

        return BertGanModelOutput(*classified)
