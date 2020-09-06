from transformers import BertConfig


class LiteralTrainPipeline:
    def __init__(self, rank, world_size, experiment, data, shared, lock):
        self.rank = rank
        self.world_size = world_size
        self.experiment = experiment
        self.data = data
        self.shared = shared
        self.lock = lock

    def __call__(self):
        labels = self.experiment.labels
        config = self.experiment.dataset.config

        bert_config = BertConfig.from_pretrained(config.model)
        bert_config.num_labels = len(labels) + 1

        data_literal = self.data.clone().literal
        data_resource = self.data.clone().resource
        data_literal.cap(config.data_size_cap)

        model = self.experiment.dataset.classifier.from_pretrained(config.model, config=bert_config)
        train = self.experiment.dataset.trainer(self.rank, self.world_size, self.experiment, model, data_literal, labels, config,
                                                self.shared, self.lock, data_neg=data_resource)

        status = f'GPU #{self.rank}: Processing category/literal classification\n'
        status += f'.. Type count: {len(labels)}\n'
        status += f'.. Data size: {data_literal.size} of {self.data.size}'

        if config.data_size_cap is not None:
            status += f' (Capped data size: {data_literal.size})'

        print(status)

        train().evaluate().save()
