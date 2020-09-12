from transformers import DistilBertModel, DistilBertPreTrainedModel

from smart.models.base import *


class DistilBertModelMixin(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

    def _forward(self, text_ids, text_masks):
        distilbert_output = self.distilbert(input_ids=text_ids, attention_mask=text_masks)
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        return self.dropout(pooled_output)


class DistilBertGanModelMixin(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        super().init_weights()

    def _forward(self, text_ids, text_masks):
        distilbert_output = self.distilbert(input_ids=text_ids, attention_mask=text_masks)
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        return self.dropout(pooled_output)


class DistilBertForMultipleLabelClassification(DistilBertModelMixin, BertForMultipleLabelClassificationBase):
    def __init__(self, config):
        super().__init__(config)
        self.init_weights()


class DistilBertForPairedBinaryClassification(DistilBertModelMixin, BertForPairedBinaryClassificationBase):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Linear(config.dim * 2, 1)
        self.init_weights()


class DistilBertGanForMultipleLabelClassification(DistilBertGanModelMixin, BertGanForMultipleLabelClassificationBase):
    def __init__(self, config):
        super().__init__(config)


class DistilBertGanForPairedBinaryClassification(DistilBertGanModelMixin, BertGanForPairedBinaryClassificationBase):
    def __init__(self, config):
        super().__init__(config)


class DistilBertGanForSequenceClassification(DistilBertGanModelMixin, BertGanForSequenceClassificationBase):
    def __init__(self, config):
        super().__init__(config)
