from transformers import BertModel, BertPreTrainedModel

from smart.models.base import *


class BertModelMixin(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        super().init_weights()

    def _forward(self, text_ids, text_masks):
        pooled_output = self.bert(input_ids=text_ids, attention_mask=text_masks)[1]
        return self.dropout(pooled_output)


class BertForMultipleLabelClassification(BertModelMixin, BertForMultipleLabelClassificationBase):
    def __init__(self, config):
        super().__init__(config)


class BertForPairedBinaryClassification(BertModelMixin, BertForPairedBinaryClassificationBase):
    def __init__(self, config):
        super().__init__(config)


class BertGanForMultipleLabelClassification(BertModelMixin, BertGanForMultipleLabelClassificationBase):
    def __init__(self, config):
        super().__init__(config)


class BertGanForPairedBinaryClassification(BertModelMixin, BertGanForPairedBinaryClassificationBase):
    def __init__(self, config):
        super().__init__(config)


class BertGanForSequenceClassification(BertModelMixin, BertGanForSequenceClassificationBase):
    def __init__(self, config):
        super().__init__(config)
