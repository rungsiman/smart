import os

from smart.experiments.base import ExperimentConfigBase, ConfigBase, TrainConfigBase
from smart.utils.configs import select


class LiteralExperimentConfig(ExperimentConfigBase):
    version = '0.12'
    experiment = 'mango-literal'
    identifier = 'base'
    description = 'Experiments run by Mango'

    class Paths(ConfigBase):
        root = 'data'

        def __init__(self, experiment, identifier):
            self.input = os.path.join(self.root, 'input')
            self.output = os.path.join(self.root, f'intermediate/literal/{experiment}-{identifier}')
            super().__init__()

            LiteralExperimentConfig.prepare(self.output)

    class Dataset(ExperimentConfigBase.Dataset):
        def __init__(self, paths, **kwargs):
            self.config = TrainConfigBase(trainer='multiple_label',
                                          labels=('boolean', 'string', 'date', 'number'),
                                          **select(kwargs, 'base-all'))

            super().__init__(paths, **select(kwargs, 'dataset'))

    class DBpedia(Dataset):
        name = 'dbpedia'

        def __init__(self, paths, **kwargs):
            self.input_root = os.path.join(paths.input, self.name)
            self.input_train = os.path.join(self.input_root, 'smarttask_dbpedia_train.json')
            self.input_test = os.path.join(self.input_root, 'smarttask_dbpedia_test_questions.json')
            self.input_ontology = os.path.join(self.input_root, 'dbpedia_types.tsv')
            super().__init__(paths, **kwargs)

    class Wikidata(Dataset):
        name = 'wikidata'

        def __init__(self, paths, **kwargs):
            self.input_root = os.path.join(paths.input, self.name)
            self.input_train = os.path.join(self.input_root, 'lcquad2_anstype_wikidata_train.json')
            self.input_test = os.path.join(self.input_root, 'lcquad2_anstype_wikidata_test.json')
            self.input_ontology = os.path.join(self.input_root, 'wikidata_types.tsv')
            super().__init__(paths, **kwargs)

    def __init__(self, dataset, **kwargs):
        self.paths = LiteralExperimentConfig.Paths(self.experiment, self.identifier)

        if dataset == 'dbpedia':
            self.dataset = LiteralExperimentConfig.DBpedia(self.paths,
                                                           **select(kwargs, 'all', 'all-literal', 'dbpedia-all', 'dbpedia-literal',))
        else:
            self.dataset = LiteralExperimentConfig.Wikidata(self.paths,
                                                            **select(kwargs, 'all', 'all-literal', 'wikidata-all', 'wikidata-literal'))

        # Apply to sklearn.model_selection.train_test_split.
        # Controls the shuffling applied to the data before applying the split.
        self.split_random_state = self.seed

        super().__init__(**select(kwargs, 'experiment-literal-base'))
