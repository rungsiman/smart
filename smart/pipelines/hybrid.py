import json
import os

from transformers import BertConfig


class HybridTrainPipeline:
    def __init__(self, rank, world_size, experiment, data, shared, lock):
        self.rank = rank
        self.world_size = world_size
        self.experiment = experiment
        self.data = data
        self.shared = shared
        self.lock = lock

    def __call__(self):
        pipeline_records = []
        self.data = self.data.resource

        for level in range(self.data.ontology.max_level):
            if level < len(self.experiment.dataset.hybrid):
                hybrid = self.experiment.dataset.hybrid[level]
            else:
                hybrid = self.experiment.dataset.hybrid_default_config

            labels = hybrid.labels
            reversed_labels = self.data.ontology.reverse(labels, level + 1)
            config = hybrid.primary_config

            bert_config = BertConfig.from_pretrained(config.model)
            bert_config.num_labels = len(labels) + 1

            # For secondary data, if filter only with (labels, reverse=True), the rest of the data on all levels will still be included.
            # If filter only with (reversed_labels), questions having both primary and secondary types will be included.
            data_primary = self.data.clone().filter(labels)
            data_secondary = self.data.clone().filter(labels, reverse=True).filter(reversed_labels)

            pipeline_records.append({'primary': {'data': data_primary.size, 'labels': len(labels)},
                                     'secondary': {'data': data_secondary.size, 'labels': len(reversed_labels)}})

            if len(labels) and data_primary.size > 0:
                data_primary.cap(config.data_size_cap)
                model = hybrid.primary_classifier.from_pretrained(config.model, config=bert_config)
                train = hybrid.primary_trainer(self.rank, self.world_size, self.experiment, model, data_primary, labels,
                                               config, self.shared, self.lock, level=level, data_neg=data_secondary)
                self._train('primary', level, train, data_primary, labels, config)

            elif self.rank == 0:
                print(f'GPU #{self.rank}: Skipped primary classification on level {level + 1}')

            if data_secondary.size > 0:
                config = hybrid.secondary_config
                data_secondary.cap(config.data_size_cap)
                model = hybrid.secondary_classifier.from_pretrained(config.model, config=bert_config)
                train = hybrid.secondary_trainer(self.rank, self.world_size, self.experiment, model, data_secondary,
                                                 config, self.shared, self.lock, level=level)
                self._train('secondary', level, train, data_secondary, reversed_labels, config)

            elif self.rank == 0:
                print(f'GPU #{self.rank}: Skipped secondary classification on level {level + 1}')

        json.dump(pipeline_records, open(os.path.join(self.experiment.dataset.output_analyses, 'pipeline_records.json'), 'w'), indent=4)

    def _train(self, identifier, level, train, data, labels, config):
        status = f'GPU #{self.rank}: Processing {identifier} classification on level {level + 1}\n'
        status += f'.. Type count: {len(labels)}\n'
        status += f'.. Data size: {data.size} of {self.data.size}'

        if config.data_size_cap is not None:
            status += f' (Data cap applied)'

        if self.rank == 0:
            print(status)

        train().evaluate().save()
