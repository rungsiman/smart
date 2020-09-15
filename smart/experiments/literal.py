import os

from smart.experiments.base import ExperimentConfigBase, ConfigBase, TrainConfigBase


class LiteralExperimentConfig(ExperimentConfigBase):
    version = '0.7-aws'
    experiment = 'distilbert-literal'
    identifier = 'sandbox'
    description = 'Sandbox for testing on AWS'

    class Paths(ConfigBase):
        root = 'data'

        def __init__(self, experiment, identifier):
            super().__init__()
            self.input = os.path.join(self.root, 'input')
            self.output = os.path.join(self.root, f'intermediate/literal/{experiment}-{identifier}')

            LiteralExperimentConfig.prepare(self.output)

    class Dataset(ExperimentConfigBase.Dataset):
        def __init__(self, paths, *args, **kwargs):
            super().__init__(paths, *args, **kwargs)
            self.config = TrainConfigBase(trainer='multiple_label',
                                          labels=('boolean', 'string', 'date', 'number'))

    class DBpedia(Dataset):
        name = 'dbpedia'

        def __init__(self, paths):
            super().__init__(paths)

            self.input_root = os.path.join(paths.input, self.name)
            self.input_train = os.path.join(self.input_root, 'smarttask_dbpedia_train.json')
            self.input_test = os.path.join(self.input_root, 'smarttask_dbpedia_test_questions.json')
            self.input_ontology = os.path.join(self.input_root, 'dbpedia_types.tsv')

    class Wikidata(Dataset):
        name = 'wikidata'

        def __init__(self, paths):
            super().__init__(paths)

            self.input_root = os.path.join(paths.input, self.name)
            self.input_train = os.path.join(self.input_root, 'lcquad2_anstype_wikidata_train.json')
            self.input_test = os.path.join(self.input_root, 'lcquad2_anstype_wikidata_test.json')
            self.input_ontology = os.path.join(self.input_root, 'wikidata_types.tsv')

    def __init__(self, dataset):
        super().__init__()

        self.paths = LiteralExperimentConfig.Paths(self.experiment, self.identifier)
        self.dataset = LiteralExperimentConfig.DBpedia(self.paths) if dataset == 'dbpedia' else LiteralExperimentConfig.Wikidata(self.paths)

        # Apply to sklearn.model_selection.train_test_split.
        # Controls the shuffling applied to the data before applying the split.
        self.split_random_state = self.seed
