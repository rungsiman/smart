import torch.distributed as dist
from transformers import BertConfig

from smart.models.bert import BertForMultipleLabelClassification, BertForPairedBinaryClassification
from smart.train.multiple_label import TrainMultipleLabelClassification
from smart.train.paired_binary import TrainPairedBinaryClassification


class DeepHybridTrain:
    def __init__(self, rank, world_size, experiment, data, shared, lock):
        self.rank = rank
        self.world_size = world_size
        self.experiment = experiment
        self.data = data
        self.shared = shared
        self.lock = lock

    def __call__(self):
        for level, deep in enumerate(self.experiment.task.deep):
            labels = deep.labels
            reversed_labels = self.data.ontology.reverse(labels, level + 1)
            config = deep.multiple_label

            bert_config = BertConfig.from_pretrained(config.model)
            bert_config.num_labels = len(labels) + 1

            data_multiple_label = self.data.clone().filter(labels)
            data_paired_binary = self.data.clone().filter(reversed_labels)

            if len(labels):
                data_multiple_label.cap(config.data_size_cap)
                model = BertForMultipleLabelClassification.from_pretrained(config.model, config=bert_config)
                train = TrainMultipleLabelClassification(self.rank, self.world_size, level, self.experiment, model, data_multiple_label, labels,
                                                         config, self.shared, self.lock, data_paired_binary)
                self._train('multiple-label', level, train, data_multiple_label, labels, config)

            data_paired_binary.cap(config.data_size_cap)
            model = BertForPairedBinaryClassification.from_pretrained(config.model, config=bert_config)
            train = TrainPairedBinaryClassification(self.rank, self.world_size, level, self.experiment, model, data_paired_binary,
                                                    config, self.shared, self.lock)
            self._train('paired-binary', level, train, data_multiple_label, reversed_labels, config)

    def _train(self, identifier, level, train, data, labels, config):
        status = f'GPU #{self.rank}: Processing {identifier} classification on level {level + 1}\n'
        status += f'.. Type count: {len(labels)}\n'
        status += f'.. Data size: {data.size} of {self.data.size}'

        if config.data_size_cap is not None:
            status += f' (Capped data size: {data.size})'

        print(status)

        dist.barrier()
        train().evaluate().save()
        dist.barrier()
