import os
from transformers import BertForSequenceClassification

from smart.experiments.base import ExperimentConfigBase, ConfigBase, BertModelConfigBase
from smart.train.sequence import TrainSequenceClassification


class LiteralExperimentConfig(ExperimentConfigBase):
    version = '0.4-aws'
    experiment = 'bert-literal'
    identifier = 'sandbox'
    description = 'Sandbox for testing on AWS'

    labels = ('boolean', 'string', 'date', 'number')

    class Paths(ConfigBase):
        root = 'data'

        def __init__(self, experiment, identifier):
            super().__init__()
            self.input = os.path.join(self.root, 'input')
            self.output = os.path.join(self.root, f'intermediate/literal/{experiment}-{identifier}')

            LiteralExperimentConfig.prepare(self.output)

    class DBpedia(ConfigBase):
        name = 'dbpedia'

        def __init__(self, paths):
            super().__init__()
            self.input_root = os.path.join(paths.input, self.name)
            self.input_train = os.path.join(self.input_root, 'smarttask_dbpedia_train.json')
            self.input_ontology = os.path.join(self.input_root, 'dbpedia_types.tsv')

            self.output_root = os.path.join(paths.output, self.name)
            self.output_train = os.path.join(self.output_root, 'train')
            self.output_test = os.path.join(self.output_root, 'test')
            self.output_models = os.path.join(self.output_root, 'models')
            self.output_analyses = os.path.join(self.output_root, 'analyses')

            LiteralExperimentConfig.prepare(self.output_root, self.output_train, self.output_test, self.output_models, self.output_analyses)

            self.config = BertModelConfigBase()
            self.classifier = BertForSequenceClassification
            self.trainer = TrainSequenceClassification

    class Wikidata(ConfigBase):
        name = 'wikidata'

        def __init__(self, paths):
            super().__init__()
            self.input_root = os.path.join(paths.input, self.name)
            self.input_train = os.path.join(self.input_root, 'lcquad2_anstype_wikidata_train.json')
            self.input_ontology = os.path.join(self.input_root, 'wikidata_types.tsv')

            self.output_root = os.path.join(paths.output, self.name)
            self.output_train = os.path.join(self.output_root, 'train')
            self.output_test = os.path.join(self.output_root, 'test')
            self.output_models = os.path.join(self.output_root, 'models')
            self.output_analyses = os.path.join(self.output_root, 'analyses')

            LiteralExperimentConfig.prepare(self.output_root, self.output_train, self.output_test, self.output_models, self.output_analyses)

            self.config = BertModelConfigBase()
            self.classifier = BertForSequenceClassification
            self.trainer = TrainSequenceClassification

    def __init__(self, dataset):
        super().__init__()
        self.paths = LiteralExperimentConfig.Paths(self.experiment, self.identifier)
        self.dataset = LiteralExperimentConfig.DBpedia(self.paths) if dataset == 'dbpedia' else LiteralExperimentConfig.Wikidata(self.paths)

        # Apply to sklearn.model_selection.train_test_split.
        # Controls the shuffling applied to the data before applying the split.
        self.split_random_state = self.seed
