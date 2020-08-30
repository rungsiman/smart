import json
import os
from datetime import datetime

from smart.experiments.base import BaseExperiment


class Experiment(BaseExperiment):
    version = '0.5-aws'
    experiment = 'bert-binary'
    identifier = 'sandbox'
    model = 'bert-base-uncased'
    description = 'Sandbox for testing on AWS'
    
    seed = 42
    epochs = 4
    batch_size = 32
    test_ratio = .1

    # IMPORTANT: For evaluating the source code on low-performance machine only.
    # Set to None for full training
    data_size = 500

    # The amount of negative examples
    neg_size = 1

    # Set to True to drop the last incomplete batch, if the dataset size is not divisible 
    # by the batch size. If False and the size of dataset is not divisible by the batch size, 
    # then the last batch will be smaller.
    drop_last = False

    # Distributed computing
    ddp_master_address = '127.0.0.1'
    ddp_master_port = '29500'

    # If set to None, the worker processes will utilize all available GPUs
    num_gpu = None
    
    class Paths(BaseExperiment.Config):
        root = 'data'

        def __init__(self, experiment, model, identifier):
            super().__init__()
            self.input = os.path.join(self.root, 'input')
            self.output = os.path.join(self.root, f'intermediate/resources/{experiment}-{identifier}/{model}')

            Experiment.prepare(self.output)
    
    class DBpedia(BaseExperiment.Config):
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

    class Bert(BaseExperiment.Config):
        learning_rate = 2e-5      # Default: 5e-5
        eps = 1e-8                # Adam's epsilon, default: 1e-8
        warmup_steps = 0
        max_grad_norm = 1.0
        
        def __init__(self):
            super().__init__()
    
    def __init__(self):
        super().__init__()
        self.paths = Experiment.Paths(self.experiment, self.model, self.identifier)
        self.task = Experiment.DBpedia(self.paths)
        self.bert = Experiment.Bert()

        # Apply to sklearn.model_selection.train_test_split.
        # Controls the shuffling applied to the data before applying the split.
        self.split_random_state = self.seed
