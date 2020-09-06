import abc
import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from smart.test.evaluation import main as ndcg_evaluate


class NDCGConfig:
    def __init__(self, experiment, path_output):
        self.type_hierarchy_tsv = experiment.dataset.input_ontology
        self.ground_truth_json = os.path.join(path_output, 'eval_truth.json')
        self.system_output_json = os.path.join(path_output, 'eval_answers.json')


class StageBase(object):
    __metaclass__ = abc.ABCMeta
    data = ...

    def _get_data(self, y_ids):
        y_dbpedia_ids = ['dbpedia_' + str(qid) for qid in y_ids]
        data = self.data.df.loc[self.data.df.id.isin(y_ids)].to_dict('records')
        data += self.data.df.loc[self.data.df.id.isin(y_dbpedia_ids)].to_dict('records')
        return data


class TrainBase(StageBase):
    path_output, path_models, path_analyses = ..., ..., ...
    train_data, train_dataloader = ..., ...
    eval_data, eval_dataloader = ..., ...
    train_records = ...
    checkpoint = ...
    sampler = ...
    identifier = ...

    def __init__(self, rank, world_size, experiment, model, data, config, shared, lock):
        self.train_records = {'loss': [], 'train_time': [], 'eval_time': None}
        self.rank = rank
        self.world_size = world_size
        self.experiment = experiment
        self.data = data
        self.config = config
        self.lock = lock
        self.shared = shared

        self.pack().build_dataloaders()
        num_training_steps = len(self.train_dataloader) * config.epochs
        model.cuda(rank)

        self.model = DistributedDataParallel(model, device_ids=[rank])
        self.optimizer = self.config.bert.optimizer.cls(model.parameters(),
                                                        *self.config.bert.optimizer.args,
                                                        **self.config.bert.optimizer.kwargs)

        if hasattr(self.optimizer, 'optimizer_'):
            self.optimizer = self.optimizer.optimizer_

        self.scheduler = self.config.bert.scheduler.cls(self.optimizer,
                                                        num_training_steps=num_training_steps,
                                                        *self.config.bert.scheduler.args,
                                                        **self.config.bert.scheduler.kwargs)
        if hasattr(self.scheduler, 'scheduler_'):
            self.scheduler = self.scheduler.scheduler_

        self.path_output = os.path.join(self.experiment.dataset.output_train, self.identifier)
        self.path_models = os.path.join(self.experiment.dataset.output_models, self.identifier)
        self.path_analyses = os.path.join(self.experiment.dataset.output_analyses, self.identifier)

        if rank == 0:
            for path in [self.path_output, self.path_models, self.path_analyses]:
                if not os.path.exists(path):
                    os.makedirs(path)

        print(f'GPU #{rank}: Initialized')

    def __call__(self):
        self.model.train()
        loss = None
        dist.barrier()

        for epoch in range(self.config.epochs):
            accumulated_loss = 0
            epoch_start = time.time()
            self.sampler.set_epoch(epoch)

            for step, batch in (enumerate(tqdm(self.train_dataloader, desc=f'GPU #{self.rank}: Training'))
                                if self.rank == 0 else enumerate(self.train_dataloader)):
                self.optimizer.zero_grad()

                loss = self._train_forward(batch)
                loss.backward()
                accumulated_loss += loss.item()

                clip_grad_norm_(parameters=self.model.parameters(),
                                max_norm=self.config.bert.max_grad_norm)

                self.optimizer.step()
                self.scheduler.step()

            average_loss = accumulated_loss / len(self.train_dataloader)
            training_time = TrainBase._format_time(time.time() - epoch_start)

            if self.rank == 0:
                status = f'GPU #{self.rank}: Training epoch {epoch + 1} of {self.config.epochs} complete:\n'
                status += f'.. Average loss: {average_loss}\n'
                status += f'.. Training time: {training_time}'
                print(status)

            self.train_records['loss'].append(average_loss)
            self.train_records['train_time'].append(training_time)

        self.checkpoint = {'epoch': self.config.epochs - 1, 'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                           'scheduler': self.scheduler.state_dict(), 'loss': loss}

        return self

    def save(self):
        if self.rank == 0:
            now = datetime.datetime.now()
            torch.save(self.checkpoint, os.path.join(self.path_models, 'model.checkpoint'))
            self.model.module.config.to_json_file(os.path.join(self.path_models, 'config.json'))
            self.data.tokenizer.save(self.path_models)
            json.dump(self.train_records, open(os.path.join(self.path_analyses, 'train_records.json'), 'w'))

            with open(os.path.join(self.path_analyses, 'train_env.txt'), 'w') as writer:
                writer.write(f'Timestamp: {now.strftime("%Y-%m-%d %H:%M:%S")}\n')
                writer.write(f'Number of GPUs: {self.world_size}')

            with open(os.path.join(self.path_analyses, 'config.json'), 'w') as writer:
                writer.write(self.experiment.describe())

            loss = np.array(self.train_records['loss'])
            plt.plot(loss, label='Training loss')
            plt.title(f'{self.experiment.experiment}-{self.experiment.identifier}-{self.identifier}')
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.legend()
            plt.savefig(os.path.join(self.path_analyses, 'loss.png'), dpi=300)
            plt.clf()

        dist.barrier()
        return self

    def build_dataloaders(self):
        self.train_dataloader = self._build_dataloader(self.train_data)
        self.eval_dataloader = self._build_dataloader(self.eval_data)
        return self

    def _save_evaluate(self, report, truths, answers):
        with open(os.path.join(self.path_analyses, 'eval_result.txt'), 'w') as writer:
            writer.write(report)

        json.dump(truths, open(os.path.join(self.path_output, 'eval_truth.json'), 'w'), indent=4)
        json.dump(answers, open(os.path.join(self.path_output, 'eval_answers.json'), 'w'), indent=4)

        ndcg_config = NDCGConfig(self.experiment, self.path_output)
        ndcg_result = ndcg_evaluate(ndcg_config)

        with open(os.path.join(self.path_analyses, 'ndcg_result.txt'), 'w') as writer:
            writer.write(ndcg_result)

        return self

    @staticmethod
    def _format_time(elapsed):
        elapsed_rounded = int(round(elapsed))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    @abc.abstractmethod
    def pack(self):
        ...

    @abc.abstractmethod
    def evaluate(self):
        ...

    @abc.abstractmethod
    def _train_forward(self, batch):
        ...

    @staticmethod
    @abc.abstractmethod
    def _build_dataloader(data):
        ...
