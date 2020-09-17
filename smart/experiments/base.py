import json
import os
from datetime import datetime
from transformers import AdamW

from smart.experiments.bert import TrainConfigMixin
from smart.utils.configs import select
from smart.utils.schedulers import LinearScheduleWithWarmup


class ConfigBase:
    """Skeleton class"""
    _obj = False

    def __init__(self, *args, **kwargs):
        self._obj = True

        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def _attrs(config):
        if str(config).startswith('<function'):
            return str(config)

        elif str(config).startswith('<class') and not getattr(config, '_obj', False):
            return str(config)

        elif any([isinstance(config, t) for t in [str, bool, int, float]]) or config is None:
            return config

        elif isinstance(config, list):
            return [ConfigBase._attrs(item) for item in config]

        elif isinstance(config, dict):
            return {key: ConfigBase._attrs(value) for key, value in config.items()}

        else:
            descriptions = {}

            for attr in dir(config):
                if attr not in ('_attrs', '_obj', 'describe', 'prepare') and not attr.startswith('__'):
                    descriptions[attr] = ConfigBase._attrs(getattr(config, attr))

            return descriptions

    def describe(self):
        return json.dumps(ConfigBase._attrs(self), indent=4)


class ClassConfigBase:
    """Skeleton class"""
    _obj = False

    def __init__(self, cls, *, args=None, kwargs=None):
        self._obj = True
        self.cls = cls
        self.args = args or []
        self.kwargs = kwargs or {}


class ExperimentConfigBase(ConfigBase):
    seed = 42

    # Distributed computing
    ddp_master_address = '127.0.0.1'
    ddp_master_port = '29500'

    # If set to None, the worker processes will utilize all available GPUs
    num_gpu = None
    main_rank = 0

    class Dataset(ConfigBase):
        name = ...

        def __init__(self, paths, *args, **kwargs):
            self.output_root = os.path.join(paths.output, self.name)
            self.output_train = os.path.join(self.output_root, 'io')
            self.output_test = os.path.join(self.output_root, 'io')
            self.output_models = os.path.join(self.output_root, 'models')
            self.output_analyses = os.path.join(self.output_root, 'analyses')

            super().__init__(*args, **kwargs)
            ExperimentConfigBase.prepare(self.output_root, self.output_train, self.output_test, self.output_models, self.output_analyses)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def prepare(*paths):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)


class BertConfigBase(ConfigBase):
    max_grad_norm = 1.0
    hidden_dropout_prob = .1

    # Dropout prob for DistilBert
    seq_classif_dropout = .2

    def __init__(self, optimizer=None, scheduler=None, *args, **kwargs):
        super().__init__(*args, **select(kwargs, 'optimizer', 'scheduler', reverse=True))

        self.optimizer = optimizer or ClassConfigBase(AdamW, kwargs={
            'lr': 2e-5,  # Default learning rate: 5e-5
            'eps': 1e-8})  # Adam's epsilon, default: 1e-6
        self.scheduler = scheduler or ClassConfigBase(LinearScheduleWithWarmup, kwargs={
            'num_warmup_steps': 0})

        if len(select(kwargs, 'optimizer')):
            self.optimizer.kwargs = {**self.optimizer.kwargs, **select(kwargs, 'optimizer')}

        if len(select(kwargs, 'scheduler')):
            self.scheduler.kwargs = {**self.scheduler.kwargs, **select(kwargs, 'scheduler')}


class GanConfigBase(ConfigBase):
    class Discriminator(BertConfigBase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class Generator(BertConfigBase):
        noise_size = 100

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        self.discriminator = GanConfigBase.Discriminator(**select(kwargs, 'discriminator'))
        self.generator = GanConfigBase.Generator(**select(kwargs, 'generator'))
        super().__init__(*args, **kwargs)


class TrainConfigBase(TrainConfigMixin, ConfigBase):
    model = 'distilbert-base-uncased'
    lowercase = True

    epochs = 4
    batch_size = 64
    eval_ratio = .1

    # IMPORTANT: For evaluating the source code on low-performance machine only.
    # Set to None for full training
    data_size_cap = None

    # The amount of negative examples
    neg_size = 1

    # When using GANs, self.bert configuration will be ignored in favor of self.gan.discriminator.
    # The discriminator's optimizer and scheduler will be applied to both BERT and the discriminator model.
    use_gan = True

    # Set to True to drop the last incomplete batch, if the dataset size is not divisible
    # by the batch size. If False and the size of dataset is not divisible by the batch size,
    # then the last batch will be smaller.
    drop_last = False

    def __init__(self, *, trainer, labels=None, **kwargs):
        self.trainer = trainer
        self.bert = BertConfigBase(**select(kwargs, 'bert'))
        self.gan = GanConfigBase(**select(kwargs, 'gan'))
        self.labels = labels
        super().__init__(**kwargs)

        if self.neg_size is not None and self.neg_size > 0:
            self.use_gan = False
        else:
            self.neg_size = 0
