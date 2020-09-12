import os

from smart.experiments.base import ExperimentConfigBase, ConfigBase, TrainConfigBase
from smart.experiments.literal import LiteralExperimentConfig
from smart.utils.hybrid import HybridConfigFactory, class_dist_thresholds


class HybridExperimentConfig(ExperimentConfigBase):
    version = '0.11-aws'
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
            super().__init__(paths, *args, **kwargs)
            self.hybrid_default = TrainConfigBase(trainer='paired_binary')

    class DBpedia(Dataset):
        name = 'dbpedia'

        def __init__(self, paths, literal):
            super().__init__(paths)

            self.input_root = os.path.join(paths.input, self.name)
            self.input_train = os.path.join(self.input_root, 'smarttask_dbpedia_train.json')
            self.input_test = os.path.join(literal.dataset.output_test, f'{literal.dataset.config.tester.resolve_identifier()}/test_answers.json')
            self.input_ontology = os.path.join(self.input_root, 'dbpedia_types.tsv')

            hybrid_factory = HybridConfigFactory(factory=class_dist_thresholds, config=TrainConfigBase, trainer='multiple_label',
                                                 input_train=self.input_train, input_ontology=self.input_ontology,
                                                 thresholds=(400,))
            self.hybrid = hybrid_factory.pack().compile()

    class Wikidata(Dataset):
        name = 'wikidata'

        def __init__(self, paths, literal):
            super().__init__(paths)

            self.input_root = os.path.join(paths.input, self.name)
            self.input_train = os.path.join(self.input_root, 'lcquad2_anstype_wikidata_train.json')
            self.input_test = os.path.join(literal.dataset.output_test, f'{literal.dataset.tester.resolve_identifier()}/test_answers.json')
            self.input_ontology = os.path.join(self.input_root, 'wikidata_types.tsv')

            hybrid_factory = HybridConfigFactory(factory=class_dist_thresholds, config=TrainConfigBase, trainer='multiple_label',
                                                 input_train=self.input_train, input_ontology=self.input_ontology,
                                                 thresholds=(400,))
            self.hybrid = hybrid_factory.pack().compile()
    
    def __init__(self, dataset):
        super().__init__()
        self.literal = LiteralExperimentConfig(dataset)
        self.paths = HybridExperimentConfig.Paths(self.experiment, self.identifier)

        if dataset == 'dbpedia':
            self.dataset = HybridExperimentConfig.DBpedia(self.paths, self.literal)
        else:
            self.dataset = HybridExperimentConfig.Wikidata(self.paths, self.literal)

        # Apply to sklearn.model_selection.train_test_split.
        # Controls the shuffling applied to the data before applying the split.
        self.split_random_state = self.seed
