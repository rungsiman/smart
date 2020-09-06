import abc
import datetime
import json
import os
from torch.nn.parallel import DistributedDataParallel

from smart.train.base import StageBase


class NDCGConfig:
    def __init__(self, experiment, path_output):
        self.type_hierarchy_tsv = experiment.dataset.input_ontology
        self.ground_truth_json = os.path.join(path_output, 'test_truth.json')
        self.system_output_json = os.path.join(path_output, 'test_answers.json')


class TestBase(StageBase):
    path_output = ...
    test_data, test_dataloader = ..., ...
    test_records = ...
    identifier = ...

    def __init__(self, rank, world_size, experiment, model, data, config, shared, lock):
        self.test_records = {'test_time': None}
        self.rank = rank
        self.world_size = world_size
        self.experiment = experiment
        self.data = data
        self.config = config
        self.lock = lock
        self.shared = shared

        self.pack().build_dataloaders()
        model.cuda(rank)

        self.model = DistributedDataParallel(model, device_ids=[rank])

        self.path_output = os.path.join(self.experiment.dataset.output_test, self.identifier)
        self.path_analyses = os.path.join(self.experiment.dataset.output_analyses, self.identifier)

        if rank == 0:
            for path in [self.path_output, self.path_analyses]:
                if not os.path.exists(path):
                    os.makedirs(path)

    def build_dataloaders(self):
        self.test_dataloader = self._build_dataloader(self.test_data)
        return self

    def save(self):
        if self.rank == 0:
            now = datetime.datetime.now()
            json.dump(self.test_records, open(os.path.join(self.path_analyses, 'test_records.json'), 'w'))

    def _save_evaluate(self, answers):
        json.dump(answers, open(os.path.join(self.path_output, 'eval_answers.json'), 'w'), indent=4)
        return self

    @abc.abstractmethod
    def pack(self):
        ...

    @staticmethod
    def _format_time(elapsed):
        elapsed_rounded = int(round(elapsed))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    @staticmethod
    @abc.abstractmethod
    def _build_dataloader(data):
        ...
