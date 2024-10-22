import abc
import datetime
import json
import os
import torch
from torch.nn.parallel import DistributedDataParallel

from smart.train.base import StageBase


class NDCGConfig:
    def __init__(self, experiment, path_output):
        self.type_hierarchy_tsv = experiment.dataset.input_ontology
        self.ground_truth_json = os.path.join(path_output, 'test_truth.json')
        self.system_output_json = os.path.join(path_output, 'test_answers.json')


class TestBase(StageBase):
    skipped = False
    path_output = ...
    test_data, test_dataloader = ..., ...
    test_records = ...
    identifier = ...
    pred_size = ...
    answers = ...

    def __init__(self, rank, world_size, experiment, model, data, labels, config, shared, lock, level=None, index=None, *args, **kwargs):
        super().__init__()

        self.test_records = {'test_time': None}
        self.rank = rank
        self.world_size = world_size
        self.experiment = experiment
        self.data = data
        self.labels = labels
        self.config = config
        self.shared = shared
        self.lock = lock
        self.level = level

        self.path_output = os.path.join(self.experiment.dataset.output_test, self.identifier)
        self.path_models = os.path.join(self.experiment.dataset.output_models, 'paired-binary' if index == 'default' else self.identifier)
        self.path_analyses = os.path.join(self.experiment.dataset.output_analyses, self.identifier)

        self.pack().build_dataloaders()
        model.cuda(rank)

        checkpoint = torch.load(os.path.join(self.path_models, 'model.checkpoint'),
                                map_location={'cuda:%d' % 0: 'cuda:%d' % rank})
        self.model = DistributedDataParallel(model, device_ids=[rank])
        self.model.load_state_dict(checkpoint['model'])

        if rank == self.experiment.main_rank:
            for path in [self.path_output, self.path_analyses]:
                if not os.path.exists(path):
                    os.makedirs(path)

    def build_dataloaders(self):
        if self.skipped:
            return self

        self.test_dataloader = self._build_dataloader(self.test_data)
        return self

    def save(self):
        if self.skipped:
            return self

        if self.rank == self.experiment.main_rank:
            now = datetime.datetime.now()
            json.dump(self.test_records, open(os.path.join(self.path_analyses, 'test_records.json'), 'w'))
            self.data.save(os.path.join(self.path_output, 'test_answers.json'))

            with open(os.path.join(self.path_analyses, 'train_records.txt'), 'w') as writer:
                writer.write(f'Timestamp: {now.strftime("%Y-%m-%d %H:%M:%S")}\n')
                writer.write(f'Number of GPUs: {self.world_size}')
                writer.write(f'Prediction size: {self.pred_size}')

        return self

    @staticmethod
    def _format_time(elapsed):
        elapsed_rounded = int(round(elapsed))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    @abc.abstractmethod
    def pack(self):
        ...

    @staticmethod
    @abc.abstractmethod
    def _build_dataloader(data):
        ...
