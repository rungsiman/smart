from transformers import BertConfig


class DeepHybridTrain:
    def __init__(self, rank, world_size, experiment, data, shared, lock):
        self.rank = rank
        self.world_size = world_size
        self.experiment = experiment
        self.data = data
        self.shared = shared
        self.lock = lock

    def __call__(self):
        for level in range(self.data.ontology.max_level):
            deep = self.experiment.task.deep[level] if level < len(self.experiment.task.deep) else self.experiment.task.deep_default_config
            labels = deep.labels
            reversed_labels = self.data.ontology.reverse(labels, level + 1)
            config = deep.primary_config

            bert_config = BertConfig.from_pretrained(config.model)
            bert_config.num_labels = len(labels) + 1

            data_multiple_label = self.data.clone().filter(labels)
            data_paired_binary = self.data.clone().filter(reversed_labels)

            if len(labels) and data_multiple_label.size > 0:
                data_multiple_label.cap(config.data_size_cap)
                model = deep.primary_classifier.from_pretrained(config.model, config=bert_config)
                train = deep.primary_trainer(self.rank, self.world_size, self.experiment, model, data_multiple_label, labels,
                                             config, self.shared, self.lock, level=level, data_neg=data_paired_binary)
                self._train('multiple-label', level, train, data_multiple_label, labels, config)
            elif self.rank == 0:
                print(f'GPU #{self.rank}: Skipped multiple-label classification on level {level + 1}')

            if data_paired_binary.size > 0:
                config = deep.secondary_config
                data_paired_binary.cap(config.data_size_cap)
                model = deep.secondary_classifier.from_pretrained(config.model, config=bert_config)
                train = deep.secondary_trainer(self.rank, self.world_size, self.experiment, model, data_paired_binary,
                                               config, self.shared, self.lock, level=level)
                self._train('paired-binary', level, train, data_multiple_label, reversed_labels, config)
            elif self.rank == 0:
                print(f'GPU #{self.rank}: Skipped paired-binary classification on level {level + 1}')

    def _train(self, identifier, level, train, data, labels, config):
        status = f'GPU #{self.rank}: Processing {identifier} classification on level {level + 1}\n'
        status += f'.. Type count: {len(labels)}\n'
        status += f'.. Data size: {data.size} of {self.data.size}'

        if config.data_size_cap is not None:
            status += f' (Capped data size: {data.size})'

        if self.rank == 0:
            print(status)

        train().evaluate().save()
