import os

from smart.models.bert import BertForMultipleLabelClassification as BML
from smart.models.bert import BertForPairedBinaryClassification as BPB
from smart.experiments.base import BaseExperiment, BertExperimentConfig as BEC
from smart.train.multiple_label import TrainMultipleLabelClassification as TML
from smart.train.paired_binary import TrainPairedBinaryClassification as TPB


class DeepConfig:
    def __init__(self, labels, primary=None, secondary=None):
        self.labels = labels
        self.primary_config, self.primary_classifier, self.primary_trainer = primary if primary is not None else (BEC(), BML, TML)
        self.secondary_config, self.secondary_classifier, self.secondary_trainer = secondary if secondary is not None else (BEC(), BPB, TPB)


class HybridExperiment(BaseExperiment):
    version = '0.6-aws'
    experiment = 'bert-hybrid'
    identifier = 'sandbox'
    description = 'Sandbox for testing on AWS'
    
    class Paths(BaseExperiment.Config):
        root = 'data'

        def __init__(self, experiment, identifier):
            super().__init__()
            self.input = os.path.join(self.root, 'input')
            self.output = os.path.join(self.root, f'intermediate/hybrid/{experiment}-{identifier}')

            HybridExperiment.prepare(self.output)

    class DBpedia(BaseExperiment.Config):
        name = 'dbpedia'

        def __init__(self, paths):
            super().__init__()
            self.input_root = os.path.join(paths.input, self.name)
            self.input_train = os.path.join(self.input_root, 'smarttask_dbpedia_train.json')
            self.input_ontology = os.path.join(self.input_root, 'dbpedia_types.tsv')

            self.output_root = os.path.join(paths.output, self.name)
            self.output = os.path.join(self.output_root, 'output')
            self.output_models = os.path.join(self.output_root, 'models')
            self.output_analyses = os.path.join(self.output_root, 'analyses')

            HybridExperiment.prepare(self.output_root, self.output, self.output_models, self.output_analyses)

            self.deep_default_config = DeepConfig([])
            self.deep = [
                DeepConfig(['dbo:Place', 'dbo:Agent', 'dbo:Work']),
                DeepConfig(['dbo:Person', 'dbo:PopulatedPlace', 'dbo:Organisation']),
                DeepConfig(['dbo:Settlement', 'dbo:Country', 'dbo:State', 'dbo:Company']),
                DeepConfig(['dbo:City', 'dbo:University', 'dbo:Stream']),
                DeepConfig(['dbo:River'])]
    
    def __init__(self):
        super().__init__()
        self.paths = HybridExperiment.Paths(self.experiment, self.identifier)
        self.task = HybridExperiment.DBpedia(self.paths)

        # Apply to sklearn.model_selection.train_test_split.
        # Controls the shuffling applied to the data before applying the split.
        self.split_random_state = self.seed
