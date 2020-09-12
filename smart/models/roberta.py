from transformers import BertPreTrainedModel, RobertaModel, RobertaConfig

from smart.models.base import *


class RobertaModelMixin(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        super().init_weights()

    def _forward(self, text_ids, text_masks):
        pooled_output = self.roberta(input_ids=text_ids, attention_mask=text_masks)[1]
        return self.dropout(pooled_output)


class RobertaForMultipleLabelClassification(RobertaModelMixin, BertForMultipleLabelClassificationBase):
    def __init__(self, config):
        super().__init__(config)


class RobertaForPairedBinaryClassification(RobertaModelMixin, BertForPairedBinaryClassificationBase):
    def __init__(self, config):
        super().__init__(config)


class RobertaGanForMultipleLabelClassification(RobertaModelMixin, BertGanForMultipleLabelClassificationBase):
    def __init__(self, config):
        super().__init__(config)


class RobertaGanForPairedBinaryClassification(RobertaModelMixin, BertGanForPairedBinaryClassificationBase):
    def __init__(self, config):
        super().__init__(config)


class RobertaGanForSequenceClassification(RobertaModelMixin, BertGanForSequenceClassificationBase):
    def __init__(self, config):
        super().__init__(config)
