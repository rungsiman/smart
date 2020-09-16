import os

from smart.experiments.base import ExperimentConfigBase, ConfigBase, TrainConfigBase
from smart.experiments.literal import LiteralExperimentConfig
from smart.utils.configs import select
from smart.utils.hybrid import HybridConfigFactory, class_dist_thresholds


class HybridExperimentConfig(ExperimentConfigBase):
    version = '0.12-aws'
    experiment = 'distilbert-hybrid'
    identifier = 'sandbox'
    description = 'Sandbox for testing on AWS'
    
    class Paths(ConfigBase):
        root = 'data'

        def __init__(self, experiment, identifier):
            super().__init__()
            self.input = os.path.join(self.root, 'input')
            self.output = os.path.join(self.root, f'intermediate/hybrid/{experiment}-{identifier}')
            self.final = os.path.join(self.root, 'final')

            HybridExperimentConfig.prepare(self.output, self.final)

    class Dataset(ExperimentConfigBase.Dataset):
        def __init__(self, paths, *args, **kwargs):
            super().__init__(paths, *args, **select(kwargs, 'dataset'))
            self.hybrid_default = TrainConfigBase(trainer='paired_binary', **kwargs)

    class DBpedia(Dataset):
        name = 'dbpedia'

        def __init__(self, paths, literal, *args, **kwargs):
            super().__init__(paths, *args, **kwargs)

            self.input_root = os.path.join(paths.input, self.name)
            self.input_train = os.path.join(self.input_root, 'smarttask_dbpedia_train.json')
            self.input_test = os.path.join(literal.dataset.output_test, f'{literal.dataset.config.tester.resolve_identifier()}/test_answers.json')
            self.input_ontology = os.path.join(self.input_root, 'dbpedia_types.tsv')

            hybrid_factory = HybridConfigFactory(factory=class_dist_thresholds, config=TrainConfigBase, trainer='multiple_label',
                                                 input_train=self.input_train, input_ontology=self.input_ontology,
                                                 thresholds=kwargs.get('class_dist_thresholds', (400,)), **kwargs)
            self.hybrid = hybrid_factory.pack().compile()

    class Wikidata(Dataset):
        name = 'wikidata'

        def __init__(self, paths, literal, *args, **kwargs):
            super().__init__(paths, *args, **kwargs)

            self.input_root = os.path.join(paths.input, self.name)
            self.input_train = os.path.join(self.input_root, 'lcquad2_anstype_wikidata_train.json')
            self.input_test = os.path.join(literal.dataset.output_test, f'{literal.dataset.tester.resolve_identifier()}/test_answers.json')
            self.input_ontology = os.path.join(self.input_root, 'wikidata_types.tsv')

            hybrid_factory = HybridConfigFactory(factory=class_dist_thresholds, config=TrainConfigBase, trainer='multiple_label',
                                                 input_train=self.input_train, input_ontology=self.input_ontology,
                                                 thresholds=kwargs.get('class_dist_thresholds', (400,)), **kwargs)
            self.hybrid = hybrid_factory.pack().compile()
    
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(*args, **select(kwargs, 'experiment-base'))
        self.literal = LiteralExperimentConfig(dataset, *args, **select(kwargs, 'experiment-base-literal'))
        self.paths = HybridExperimentConfig.Paths(self.experiment, self.identifier)

        if dataset == 'dbpedia':
            self.dataset = HybridExperimentConfig.DBpedia(self.paths, self.literal, **select(kwargs, 'train-base', 'test-base'))
        else:
            self.dataset = HybridExperimentConfig.Wikidata(self.paths, self.literal, **select(kwargs, 'train-base', 'test-base'))

        # Apply to sklearn.model_selection.train_test_split.
        # Controls the shuffling applied to the data before applying the split.
        self.split_random_state = self.seed
