import os

from smart.experiments.base import ExperimentConfigBase, ConfigBase, TrainConfigBase
from smart.experiments.literal import LiteralExperimentConfig
from smart.utils.configs import select
from smart.utils.hybrid import HybridConfigFactory, class_dist_thresholds


class HybridExperimentConfig(ExperimentConfigBase):
    version = '0.18'
    experiment = 'mango-hybrid'
    identifier = 'base'
    description = 'Experiments run by Mango'
    
    class Paths(ConfigBase):
        root = 'data'

        def __init__(self, experiment, identifier, **kwargs):
            self.input = os.path.join(self.root, 'input')
            self.output = os.path.join(self.root, f'intermediate/hybrid/{experiment}-{identifier}')
            self.final = os.path.join(self.root, 'final')
            super().__init__(**kwargs)

            HybridExperimentConfig.prepare(self.output, self.final)

    class Dataset(ExperimentConfigBase.Dataset):
        def __init__(self, paths, **kwargs):
            # Available options for test strategy:
            # .. dependent: the decision to make predictions on a deeper level depends on the predictions on a higher level
            # .. independent: the decision does not depend on other levels
            # .. top-down: predict independently but remove predictions on lower level for classes having no parents
            # .. bottom-up: predict independently and add parents to classes with no parents
            # Paired-label classification in all independent strategies will still be dependent to avoid excessive computation requirement,
            # unless 'independent_paired_label' is true.
            self.test_strategy = 'dependent'
            self.independent_paired_label = False

            print('>>> ' + str(kwargs))

            if kwargs.get('skip_default_classifier', False):
                self.hybrid_default = None
            else:
                self.hybrid_default = TrainConfigBase(trainer='paired_binary', **select(kwargs, 'base-all', 'base-default'))

            super().__init__(paths, **select(kwargs, 'dataset'))

    class DBpedia(Dataset):
        name = 'dbpedia'

        def __init__(self, paths, literal, **kwargs):
            self.input_root = os.path.join(paths.input, self.name)
            self.input_train = os.path.join(self.input_root, 'smarttask_dbpedia_train.json')
            self.input_test = os.path.join(literal.dataset.output_test, f'{literal.dataset.config.tester.resolve_identifier()}/test_answers.json')
            self.input_ontology = os.path.join(self.input_root, 'dbpedia_types.tsv')

            hybrid_factory = HybridConfigFactory(factory=class_dist_thresholds, config=TrainConfigBase, trainer='multiple_label',
                                                 input_train=self.input_train, input_ontology=self.input_ontology,
                                                 thresholds=kwargs.get('class_dist_thresholds', (400,)), **select(kwargs, 'base'))
            self.hybrid = hybrid_factory.pack().compile()
            self.selective_train = None

            super().__init__(paths, **kwargs)

    class Wikidata(Dataset):
        name = 'wikidata'

        def __init__(self, paths, literal, **kwargs):
            self.input_root = os.path.join(paths.input, self.name)
            self.input_train = os.path.join(self.input_root, 'lcquad2_anstype_wikidata_train.json')
            self.input_test = os.path.join(literal.dataset.output_test, f'{literal.dataset.config.tester.resolve_identifier()}/test_answers.json')
            self.input_ontology = os.path.join(self.input_root, 'wikidata_types.tsv')

            hybrid_factory = HybridConfigFactory(factory=class_dist_thresholds, config=TrainConfigBase, trainer='multiple_label',
                                                 input_train=self.input_train, input_ontology=self.input_ontology,
                                                 thresholds=kwargs.get('class_dist_thresholds', (400,)), **select(kwargs, 'base'))
            self.hybrid = hybrid_factory.pack().compile()
            self.selective_train = None

            super().__init__(paths, **kwargs)
    
    def __init__(self, dataset, **kwargs):
        self.literal = LiteralExperimentConfig(dataset, **select(kwargs, 'experiment-hybrid-literal-base'))
        self.paths = HybridExperimentConfig.Paths(self.experiment, self.identifier)

        if dataset == 'dbpedia':
            self.dataset = HybridExperimentConfig.DBpedia(self.paths, self.literal,
                                                          **select(kwargs, 'all', 'all-hybrid', 'dbpedia-all', 'dbpedia-hybrid'))
        else:
            self.dataset = HybridExperimentConfig.Wikidata(self.paths, self.literal,
                                                           **select(kwargs, 'all', 'all-hybrid', 'wikidata-all', 'wikidata-hybrid'))

        # Apply to sklearn.model_selection.train_test_split.
        # Controls the shuffling applied to the data before applying the split.
        self.split_random_state = self.seed

        super().__init__(**select(kwargs, 'experiment-hybrid-base'))
