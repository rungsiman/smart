import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import PreTrainedModel

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


class BertForMultipleLabelClassificationBase(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, text_ids, text_masks, labels=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pooled_output = self._forward(text_ids, text_masks)
        logits = self.classifier(pooled_output)
        loss = None

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.float().view(-1, self.num_labels))

        if return_dict is None:
            return loss if labels is not None else logits

        return BertModelOutput(loss, logits)


class BertForPairedBinaryClassificationBase(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.classifier = nn.Linear(config.hidden_size * 2, 1)

    def forward(self, text_ids, text_masks, label_ids, label_masks, labels=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pooled_text_output = self._forward(text_ids, text_masks)
        pooled_label_output = self._forward(label_ids, label_masks)
        pooled_output = torch.cat((pooled_text_output, pooled_label_output), dim=1)

        logits = self.classifier(pooled_output)
        loss = None

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float().view(-1))

        if return_dict is None:
            return loss if labels is not None else logits

        return BertModelOutput(loss, logits)


class BertGanForMultipleLabelClassificationBase(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        config.gan_hidden_size = getattr(config, 'dim', config.hidden_size)
        self.classifier = GanForMultipleLabelClassification(config)

    def forward(self, text_ids, text_masks, labels=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pooled_output = self._forward(text_ids, text_masks)
        classified = self.classifier(pooled_output, labels)

        if return_dict is None:
            return classified

        return BertGanModelOutput(*classified)


class BertGanForPairedBinaryClassificationBase(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        config.num_labels = 1
        config.gan_hidden_size = getattr(config, 'dim', config.hidden_size) * 2
        self.classifier = GanForPairedLabelClassification(config)

    def forward(self, text_ids, text_masks, label_ids, label_masks, labels=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pooled_text_output = self._forward(text_ids, text_masks)
        pooled_label_output = self._forward(label_ids, label_masks)
        pooled_output = torch.cat((pooled_text_output, pooled_label_output), dim=1)

        classified = self.classifier(pooled_output, labels)

        if return_dict is None:
            return classified

        return BertGanModelOutput(*classified)


class BertGanForSequenceClassificationBase(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        config.gan_hidden_size = getattr(config, 'dim', config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = GanForSequenceClassification(config)

    def forward(self, text_ids, text_masks, labels=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        features = self._forward(text_ids, text_masks)

        pooled_output = self.dropout(features)
        classified = self.classifier(pooled_output, labels)

        if return_dict is None:
            return classified

        return BertGanModelOutput(*classified)
