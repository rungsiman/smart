from transformers import BertConfig, BertForSequenceClassification
from transformers import DistilBertConfig, DistilBertForSequenceClassification
from transformers import RobertaConfig, RobertaForSequenceClassification

from smart.models.bert import *
from smart.models.roberta import *
from smart.models.distilbert import *
from smart.test.multiple_label import *
from smart.test.paired_binary import *
from smart.test.sequence import *
from smart.train.multiple_label import *
from smart.train.paired_binary import *
from smart.train.sequence import *


class TrainConfigMixin:
    model = ...
    use_gan = ...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def trainer(self):
        mapper = {'multiple_label': TrainMultipleLabelClassification,
                  'paired_binary': TrainPairedBinaryClassification,
                  'sequence': TrainSequenceClassification}
        return mapper[self.__trainer]

    @trainer.setter
    def trainer(self, val):
        self.__trainer = val

    @property
    def tester(self):
        mapper = {'multiple_label': TestMultipleLabelClassification,
                  'paired_binary': TestPairedBinaryClassification,
                  'sequence': TestSequenceClassification}
        return mapper[self.__trainer]

    @property
    def classifier(self):
        if self.use_gan:
            if self.model.startswith('bert'):
                mapper = {'multiple_label': BertGanForMultipleLabelClassification,
                          'paired_binary': BertGanForPairedBinaryClassification,
                          'sequence': BertGanForSequenceClassification}
                return mapper[self.__trainer]

            elif self.model.startswith('distilbert'):
                mapper = {'multiple_label': DistilBertGanForMultipleLabelClassification,
                          'paired_binary': DistilBertGanForPairedBinaryClassification,
                          'sequence': DistilBertGanForSequenceClassification}
                return mapper[self.__trainer]

            elif self.model.startswith('roberta'):
                mapper = {'multiple_label': RobertaGanForMultipleLabelClassification,
                          'paired_binary': RobertaGanForPairedBinaryClassification,
                          'sequence': RobertaGanForSequenceClassification}
                return mapper[self.__trainer]

        else:
            if self.model.startswith('bert'):
                mapper = {'multiple_label': BertForMultipleLabelClassification,
                          'paired_binary': BertForPairedBinaryClassification,
                          'sequence': BertForSequenceClassification}
                return mapper[self.__trainer]

            elif self.model.startswith('distilbert'):
                mapper = {'multiple_label': DistilBertForMultipleLabelClassification,
                          'paired_binary': DistilBertForPairedBinaryClassification,
                          'sequence': DistilBertForSequenceClassification}
                return mapper[self.__trainer]

            elif self.model.startswith('roberta'):
                mapper = {'multiple_label': RobertaForMultipleLabelClassification,
                          'paired_binary': RobertaForPairedBinaryClassification,
                          'sequence': RobertaForSequenceClassification}
                return mapper[self.__trainer]

    @property
    def bert_config(self):
        if self.model.startswith('bert'):
            return BertConfig
        elif self.model.startswith('distilbert'):
            return DistilBertConfig
        elif self.model.startswith('roberta'):
            return RobertaConfig
