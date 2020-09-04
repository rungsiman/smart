from transformers import BertConfig


class DeepLiteralTrain:
    def __init__(self, rank, world_size, experiment, data, shared, lock):
        self.rank = rank
        self.world_size = world_size
        self.experiment = experiment
        self.data = data
        self.shared = shared
        self.lock = lock

    def __call__(self):
        labels = ('boolean', 'string', 'date', 'number', 'resource')
        config = self.experiment.task.config

        bert_config = BertConfig.from_pretrained(config.model)
        bert_config.num_labels = len(labels)

        data = self.data.clone().filter(labels)
        data.cap(config.data_size_cap)

        model = self.experiment.task.classifier.from_pretrained(config.model, config=bert_config)
        train = self.experiment.task.trainer(self.rank, self.world_size, self.experiment, model, data, labels, config, self.shared, self.lock)

        status = f'GPU #{self.rank}: Processing category/literal classification\n'
        status += f'.. Type count: {len(labels)}\n'
        status += f'.. Data size: {data.size} of {self.data.size}'

        if config.data_size_cap is not None:
            status += f' (Capped data size: {data.size})'

        print(status)

        train().evaluate().save()
