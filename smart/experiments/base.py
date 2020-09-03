import json
import os
from datetime import datetime


class BaseExperiment:
    class Config:
        _obj = False
        
        def __init__(self):
            self._obj = True
            
    def __init__(self):
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def attrs(config):
        if any([isinstance(config, t) for t in [str, bool, int, float]]) or config is None:
            return config

        else:
            descriptions = {}

            for attr in dir(config):
                if attr != 'attrs' and attr != '_obj' and not attr.startswith('__'):
                    obj = getattr(config, attr)

                    if any([isinstance(obj, t) for t in [str, bool, int, float]]) or obj is None:
                        descriptions[attr] = getattr(config, attr)

                    elif isinstance(obj, list):
                        descriptions[attr] = [BaseExperiment.attrs(item) for item in obj]

                    elif isinstance(obj, dict):
                        descriptions[attr] = {key: BaseExperiment.attrs(value) for key, value in obj.items()}

                    else:
                        descriptions[attr] = BaseExperiment.attrs(obj)

            return descriptions
    
    def __str__(self):
        return json.dumps(BaseExperiment.attrs(self), indent=4)

    @staticmethod
    def prepare(*paths):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)


class BertConfig:
    model = 'bert-base-uncased'

    epochs = 4
    batch_size = 32
    eval_ratio = .1

    # IMPORTANT: For evaluating the source code on low-performance machine only.
    # Set to None for full training
    data_size_cap = 500

    # The amount of negative examples
    neg_size = 1

    # Set to True to drop the last incomplete batch, if the dataset size is not divisible
    # by the batch size. If False and the size of dataset is not divisible by the batch size,
    # then the last batch will be smaller.
    drop_last = False

    class Bert(BaseExperiment.Config):
        learning_rate = 2e-5  # Default: 5e-5
        eps = 1e-8  # Adam's epsilon, default: 1e-8
        warmup_steps = 0
        max_grad_norm = 1.0

        def __init__(self):
            super().__init__()

    def __init__(self):
        self.bert = BertConfig.Bert()
