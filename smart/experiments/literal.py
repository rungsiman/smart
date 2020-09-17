import os

from smart.experiments.base import ExperimentConfigBase, ConfigBase, TrainConfigBase
from smart.utils.configs import select


class LiteralExperimentConfig(ExperimentConfigBase):
    version = '0.9'
    experiment = 'cat-hybrid'
    identifier = 'base'
    description = 'Experiments run by Cat'

    class Paths(ConfigBase):
        root = 'data'

        def __init__(self, experiment, identifier):
            super().__init__()
            self.input = os.path.join(self.root, 'input')
            self.output = os.path.join(self.root, f'intermediate/literal/{experiment}-{identifier}')

            LiteralExperimentConfig.prepare(self.output)

    class Dataset(ExperimentConfigBase.Dataset):
        def __init__(self, paths, *args, **kwargs):
            super().__init__(paths, *args, **select(kwargs, 'dataset'))
            self.config = TrainConfigBase(trainer='multiple_label',
                                          labels=('boolean', 'string', 'date', 'number'),
                                          **kwargs)

    class DBpedia(Dataset):
        name = 'dbpedia'

        def __init__(self, paths, **kwargs):
            super().__init__(paths, **kwargs)

            self.input_root = os.path.join(paths.input, self.name)
            self.input_train = os.path.join(self.input_root, 'smarttask_dbpedia_train.json')
            self.input_test = os.path.join(self.input_root, 'smarttask_dbpedia_test_questions.json')
            self.input_ontology = os.path.join(self.input_root, 'dbpedia_types.tsv')

    class Wikidata(Dataset):
        name = 'wikidata'

        def __init__(self, paths, **kwargs):
            super().__init__(paths, **kwargs)

            self.input_root = os.path.join(paths.input, self.name)
            self.input_train = os.path.join(self.input_root, 'lcquad2_anstype_wikidata_train.json')
            self.input_test = os.path.join(self.input_root, 'lcquad2_anstype_wikidata_test.json')
            self.input_ontology = os.path.join(self.input_root, 'wikidata_types.tsv')

    def __init__(self, dataset, *args, **kwargs):
        super().__init__(*args, **select(kwargs, 'experiment-base-literal'))

        self.paths = LiteralExperimentConfig.Paths(self.experiment, self.identifier)

        if dataset == 'dbpedia':
            self.dataset = LiteralExperimentConfig.DBpedia(self.paths, **select(kwargs, 'train-base-literal', 'test-base-literal'))
        else:
            self.dataset = LiteralExperimentConfig.Wikidata(self.paths, **select(kwargs, 'train-base-literal', 'test-base-literal'))

        # Apply to sklearn.model_selection.train_test_split.
        # Controls the shuffling applied to the data before applying the split.
        self.split_random_state = self.seed
