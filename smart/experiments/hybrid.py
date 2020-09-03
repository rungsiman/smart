import os

from smart.experiments.base import BaseExperiment, BertConfig


class DeepConfig:
    def __init__(self, labels, multiple_label=None, paired_binary=None):
        self.labels = labels
        self.multiple_label = multiple_label or BertConfig()
        self.paired_binary = paired_binary or BertConfig()


class Experiment(BaseExperiment):
    version = '0.6-aws'
    experiment = 'bert-hybrid'
    identifier = 'sandbox'
    description = 'Sandbox for testing on AWS'

    seed = 42
    tokenizer_model = 'bert-base-uncased'
    tokenizer_lowercase = True

    # Distributed computing
    ddp_master_address = '127.0.0.1'
    ddp_master_port = '29500'

    # If set to None, the worker processes will utilize all available GPUs
    num_gpu = None
    
    class Paths(BaseExperiment.Config):
        root = 'data'

        def __init__(self, experiment, identifier):
            super().__init__()
            self.input = os.path.join(self.root, 'input')
            self.output = os.path.join(self.root, f'intermediate/hybrid/{experiment}-{identifier}')

            Experiment.prepare(self.output)

    class DBpedia(BaseExperiment.Config):
        name = 'dbpedia'

        def __init__(self, paths):
            super().__init__()
            self.input_root = os.path.join(paths.input, 'dbpedia')
            self.input_train = os.path.join(self.input_root, 'smarttask_dbpedia_train.json')
            self.input_ontology = os.path.join(self.input_root, 'dbpedia_types.tsv')

            self.output_root = os.path.join(paths.output, 'dbpedia')
            self.output = os.path.join(self.output_root, 'output')
            self.output_models = os.path.join(self.output_root, 'models')
            self.output_analyses = os.path.join(self.output_root, 'analyses')

            Experiment.prepare(self.output_root, self.output, self.output_models, self.output_analyses)

            self.deep = [
                DeepConfig(['dbo:Place', 'dbo:Agent', 'dbo:Work']),
                DeepConfig(['dbo:Person', 'dbo:PopulatedPlace', 'dbo:Organisation']),
                DeepConfig(['dbo:Settlement', 'dbo:Country', 'dbo:State', 'dbo:Company'],),
                DeepConfig(['dbo:City', 'dbo:University', 'dbo:Stream']),
                DeepConfig(['dbo:River'])]
    
    def __init__(self):
        super().__init__()
        self.paths = Experiment.Paths(self.experiment, self.identifier)
        self.task = Experiment.DBpedia(self.paths)

        # Apply to sklearn.model_selection.train_test_split.
        # Controls the shuffling applied to the data before applying the split.
        self.split_random_state = self.seed
