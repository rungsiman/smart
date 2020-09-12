import os

from smart.data.base import Ontology
from smart.data.tokenizers import CustomAutoTokenizer
from smart.pipelines.base import PipelineBase
from smart.utils.configs import override


class LiteralTrainPipeline(PipelineBase):
    def __call__(self):
        config = self.experiment.dataset.config

        bert_config = config.bert_config.from_pretrained(config.model)
        bert_config.num_labels = len(config.labels) + 1
        override(bert_config, config)

        tokenizer = CustomAutoTokenizer(config)
        ontology = Ontology(self.experiment.dataset.input_ontology).tokenize(tokenizer)
        self.data.tokenize(ontology, tokenizer)

        data_literal = self.data.clone().literal
        data_resource = self.data.clone().resource
        data_literal.cap(config.data_size_cap)

        model = config.classifier.from_pretrained(config.model, config=bert_config)
        train = config.trainer(self.rank, self.world_size, self.experiment, model, data_literal, config.labels, config,
                               self.shared, self.lock, data_neg=data_resource)

        status = f'GPU #{self.rank}: Processing category/literal training\n'
        status += f'.. Type count: {len(config.labels)}\n'
        status += f'.. Data size: {data_literal.size} of {self.data.size}'

        if config.data_size_cap is not None:
            status += f' (Capped data size: {data_literal.size})'

        print(status)

        train().evaluate().save()


class LiteralTestPipeline(PipelineBase):
    def __call__(self):
        config = self.experiment.dataset.config
        identifier = config.tester.resolve_identifier()
        path_models = os.path.join(self.experiment.dataset.output_models, identifier)

        bert_config = config.bert_config.from_pretrained(path_models)
        override(bert_config, config)

        tokenizer = CustomAutoTokenizer(config, path_models)
        ontology = Ontology(self.experiment.dataset.input_ontology).tokenize(tokenizer)
        self.data.tokenize(ontology, tokenizer)

        data_truth = self.data.clone().cap(config.data_size_cap)
        data_test = data_truth.clone().blind()

        model = config.classifier.from_pretrained(config.model, config=bert_config)
        test = config.tester(self.rank, self.world_size, self.experiment, model, data_test, config.labels, config,
                             self.shared, self.lock)

        status = f'GPU #{self.rank}: Processing category/literal testing\n'
        status += f'.. Type count: {len(config.labels)}\n'
        status += f'.. Data size: {data_test.size} of {self.data.size}'

        if config.data_size_cap is not None:
            status += f' (Data cap applied)'

        print(status)
        test()
        test.data.assign_answers(test.answers)
        test.data.assign_categories()
        test.save()

        if self.rank == 0:
            status = f'GPU #{self.rank}: Testing complete.\n'
            status += f'.. Answer count: {test.data.count_answers()}'
            print(status)
