import os

from smart.models.bert import BertForMultipleLabelClassification as BML
from smart.models.bert import BertForPairedBinaryClassification as BPB
from smart.experiments.base import ExperimentConfigBase, ConfigBase, BertModelConfig as BMC
from smart.train.multiple_label import TrainMultipleLabelClassification as TML
from smart.train.paired_binary import TrainPairedBinaryClassification as TPB


class HybridConfig:
    def __init__(self, labels, primary=None, secondary=None):
        self.labels = labels
        self.primary_config, self.primary_classifier, self.primary_trainer = primary or (BMC(), BML, TML)
        self.secondary_config, self.secondary_classifier, self.secondary_trainer = secondary or (BMC(), BPB, TPB)


class HybridExperimentConfig(ExperimentConfigBase):
    version = '0.9-aws'
    experiment = 'bert-hybrid'
    identifier = 'sandbox'
    description = 'Sandbox for testing on AWS'
    
    class Paths(ConfigBase):
        root = 'data'

        def __init__(self, experiment, identifier):
            super().__init__()
            self.input = os.path.join(self.root, 'input')
            self.output = os.path.join(self.root, f'intermediate/hybrid/{experiment}-{identifier}')

            HybridExperimentConfig.prepare(self.output)

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

            HybridExperimentConfig.prepare(self.output_root, self.output_train, self.output_test, self.output_models, self.output_analyses)

            self.hybrid_default_config = HybridConfig([])
            self.hybrid = [HybridConfig(['dbo:Place', 'dbo:Agent', 'dbo:Work']),
                           HybridConfig(['dbo:Person', 'dbo:PopulatedPlace', 'dbo:Organisation']),
                           HybridConfig(['dbo:Settlement', 'dbo:Country', 'dbo:State', 'dbo:Company']),
                           HybridConfig(['dbo:City', 'dbo:University', 'dbo:Stream']),
                           HybridConfig(['dbo:River'])]

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

            HybridExperimentConfig.prepare(self.output_root, self.output_train, self.output_test, self.output_models, self.output_analyses)

            self.hybrid_default_config = HybridConfig([])
            self.hybrid = [HybridConfig(['omnivore']),
                           HybridConfig(['natural person']),
                           HybridConfig(['person', 'state']),
                           HybridConfig(['human settlement']),
                           HybridConfig(['city/town']),
                           HybridConfig([]),
                           HybridConfig(['political territorial entity']),
                           HybridConfig(['country'])]
    
    def __init__(self, dataset):
        super().__init__()
        self.paths = HybridExperimentConfig.Paths(self.experiment, self.identifier)
        self.dataset = HybridExperimentConfig.DBpedia(self.paths) if dataset == 'dbpedia' else HybridExperimentConfig.Wikidata(self.paths)

        # Apply to sklearn.model_selection.train_test_split.
        # Controls the shuffling applied to the data before applying the split.
        self.split_random_state = self.seed
