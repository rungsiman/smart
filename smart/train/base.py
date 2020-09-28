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
    data = ...
    data_neg = None

    def _get_data(self, y_ids):
        y_dbpedia_ids = ['dbpedia_' + str(qid) for qid in y_ids]
        data = self.data.df.loc[self.data.df.id.isin(y_ids)].to_dict('records')
        data += self.data.df.loc[self.data.df.id.isin(y_dbpedia_ids)].to_dict('records')

        if self.data_neg is not None:
            data += self.data_neg.df.loc[self.data_neg.df.id.isin(y_ids)].to_dict('records')
            data += self.data_neg.df.loc[self.data_neg.df.id.isin(y_dbpedia_ids)].to_dict('records')

        return data


class TrainBase(StageBase):
    name = ...
    skipped = False
    path_output, path_models, path_analyses = ..., ..., ...
    train_data, train_dataloader = ..., ...
    eval_data, eval_dataloader = ..., ...
    eval_report, eval_dict, eval_truths, eval_answers, ndcg_result = None, None, None, None, None
    train_records = ...
    checkpoint = ...
    sampler = ...
    identifier = ...

    def __init__(self, rank, world_size, experiment, model, data, labels, config, shared, lock,
                 data_neg=None, level=None, index=None, *args, **kwargs):
        super().__init__()

        self.train_records = {'loss': [], 'train_time': [], 'eval_time': None}
        self.rank = rank
        self.world_size = world_size
        self.experiment = experiment
        self.data = data
        self.config = config
        self.lock = lock
        self.shared = shared
        self.labels = labels
        self.data_neg = data_neg
        self.level = level
        self.index = index

        self.path_output = os.path.join(self.experiment.dataset.output_train, self.identifier)
        self.path_models = os.path.join(self.experiment.dataset.output_models, self.identifier)
        self.path_analyses = os.path.join(self.experiment.dataset.output_analyses, self.identifier)

        self.pack().build_dataloaders()
        num_training_steps = len(self.train_dataloader) * config.epochs
        model.cuda(rank)

        self.model = DistributedDataParallel(model, device_ids=[rank])

        if self.config.use_gan:
            self.optimizer = TrainBase._apply_optimizer(getattr(self.model.module, self.model.module.base_model_prefix), self.config.bert.optimizer)
            self.scheduler = TrainBase._apply_scheduler(self.optimizer, self.config.bert.scheduler, num_training_steps)
            self.d_optimizer = TrainBase._apply_optimizer(self.model.module.classifier.discriminator, self.config.gan.discriminator.optimizer)
            self.g_optimizer = TrainBase._apply_optimizer(self.model.module.classifier.generator, self.config.gan.generator.optimizer)
            self.d_scheduler = TrainBase._apply_scheduler(self.d_optimizer, self.config.gan.discriminator.scheduler, num_training_steps)
            self.g_scheduler = TrainBase._apply_scheduler(self.g_optimizer, self.config.gan.generator.scheduler, num_training_steps)

        else:
            self.optimizer = TrainBase._apply_optimizer(self.model.module, self.config.bert.optimizer)
            self.scheduler = TrainBase._apply_scheduler(self.optimizer, self.config.bert.scheduler, num_training_steps)

        if rank == self.experiment.main_rank:
            for path in [self.path_output, self.path_models, self.path_analyses]:
                if not os.path.exists(path):
                    os.makedirs(path)

        print(f'GPU #{rank}: Initialized')

    def __call__(self):
        self.model.train()
        dist.barrier()

        return self.train() if not self.config.use_gan else self.train_gan()

    def train(self):
        loss = None

        for epoch in range(self.config.epochs):
            accumulated_loss = 0
            epoch_start = time.time()
            self.sampler.set_epoch(epoch)

            for step, batch in (enumerate(tqdm(self.train_dataloader, desc=f'GPU #{self.rank}: Training'))
                                if self.rank == self.experiment.main_rank else enumerate(self.train_dataloader)):
                self.optimizer.zero_grad()

                loss = self._train_forward(batch).loss
                loss.backward()
                accumulated_loss += loss.item()

                clip_grad_norm_(parameters=self.model.module.parameters(),
                                max_norm=self.config.bert.max_grad_norm)

                self.optimizer.step()
                self.scheduler.step()

            average_loss = accumulated_loss / len(self.train_dataloader)
            training_time = TrainBase._format_time(time.time() - epoch_start)

            if self.rank == self.experiment.main_rank:
                status = f'GPU #{self.rank}: Training epoch {epoch + 1} of {self.config.epochs} complete:\n'
                status += f'.. Average loss: {average_loss}\n'
                status += f'.. Training time: {training_time}'
                print(status)

            self.train_records['loss'].append(average_loss)
            self.train_records['train_time'].append(training_time)

        self.checkpoint = {'epoch': self.config.epochs - 1, 'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                           'scheduler': self.scheduler.state_dict(), 'loss': loss}

        return self

    def train_gan(self):
        d_loss, g_loss = None, None

        for epoch in range(self.config.epochs):
            accumulated_d_loss = 0
            accumulated_g_loss = 0
            epoch_start = time.time()
            self.sampler.set_epoch(epoch)

            for step, batch in (enumerate(tqdm(self.train_dataloader, desc=f'GPU #{self.rank}: Training'))
                                if self.rank == self.experiment.main_rank else enumerate(self.train_dataloader)):
                # Update BERT and discriminator's parameters with d_loss
                self.optimizer.zero_grad()
                self.d_optimizer.zero_grad()

                loss = self._train_forward(batch)
                d_loss, g_loss = loss.d_loss, loss.g_loss
                d_loss.backward(retain_graph=True)

                self.optimizer.step()
                self.scheduler.step()

                self.d_optimizer.step()
                self.d_scheduler.step()

                clip_grad_norm_(parameters=getattr(self.model.module, self.model.module.base_model_prefix).parameters(),
                                max_norm=self.config.bert.max_grad_norm)
                clip_grad_norm_(parameters=self.model.module.classifier.discriminator.parameters(),
                                max_norm=self.config.gan.discriminator.max_grad_norm)

                accumulated_d_loss += d_loss.item()

                # Update generator's parameters with g_loss
                self.g_optimizer.zero_grad()

                g_loss.backward()

                self.g_optimizer.step()
                self.g_scheduler.step()

                clip_grad_norm_(parameters=self.model.module.classifier.generator.parameters(),
                                max_norm=self.config.gan.generator.max_grad_norm)

                accumulated_g_loss += g_loss.item()

            average_d_loss = accumulated_d_loss / len(self.train_dataloader)
            average_g_loss = accumulated_g_loss / len(self.train_dataloader)
            training_time = TrainBase._format_time(time.time() - epoch_start)

            if self.rank == self.experiment.main_rank:
                status = f'GPU #{self.rank}: Training epoch {epoch + 1} of {self.config.epochs} complete:\n'
                status += f'.. Average discriminator loss: {average_d_loss}\n'
                status += f'.. Average generator loss: {average_g_loss}\n'
                status += f'.. Training time: {training_time}'
                print(status)

            self.train_records['loss'].append({'d_loss': average_d_loss, 'g_loss': average_g_loss})
            self.train_records['train_time'].append(training_time)

        self.checkpoint = {'epoch': self.config.epochs - 1, 'model': self.model.state_dict(),
                           'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(),
                           'd_optimizer': self.d_optimizer.state_dict(), 'd_scheduler': self.d_scheduler.state_dict(), 'd_loss': d_loss,
                           'g_optimizer': self.g_optimizer.state_dict(), 'g_scheduler': self.g_scheduler.state_dict(), 'g_loss': g_loss}

        return self

    def save(self):
        if self.rank == self.experiment.main_rank:
            now = datetime.datetime.now()
            torch.save(self.checkpoint, os.path.join(self.path_models, 'model.checkpoint'))
            self.model.module.config.to_json_file(os.path.join(self.path_models, 'config.json'))
            self.data.tokenizer.save(self.path_models)
            json.dump(self.train_records, open(os.path.join(self.path_analyses, 'train_records.json'), 'w'))

            with open(os.path.join(self.path_analyses, 'train_records.txt'), 'w') as writer:
                writer.write(f'Timestamp: {now.strftime("%Y-%m-%d %H:%M:%S")}\n')
                writer.write(f'Number of GPUs: {self.world_size}')

            with open(os.path.join(self.path_analyses, 'config.json'), 'w') as writer:
                writer.write(self.experiment.describe())

            chart_title = f'{self.experiment.experiment}-{self.experiment.identifier}-{self.identifier}'

            if self.config.use_gan:
                d_loss = np.array([loss['d_loss'] for loss in self.train_records['loss']])
                g_loss = np.array([loss['g_loss'] for loss in self.train_records['loss']])
                self._plot_loss(chart_title, 'loss.png', d_loss=d_loss, g_loss=g_loss)
            else:
                loss = np.array(self.train_records['loss'])
                self._plot_loss(chart_title, 'loss.png', loss=loss)

            if self.eval_report is not None:
                self._save_evaluate()
            elif self.rank == self.experiment.main_rank:
                print(f'GPU #{self.rank}: Skipped saving evaluation results.')

        dist.barrier()
        return self

    def _save_evaluate(self):
        if self.config.eval_ratio is None or self.config.eval_ratio == 0:
            return self

        with open(os.path.join(self.path_analyses, 'eval_result.txt'), 'w') as writer:
            writer.write(self.eval_report)

        json.dump(self.eval_truths, open(os.path.join(self.path_output, 'eval_truth.json'), 'w'), indent=4)
        json.dump(self.eval_answers, open(os.path.join(self.path_output, 'eval_answers.json'), 'w'), indent=4)

        if self.experiment.dataset.name == 'dbpedia':
            try:
                ndcg_config = NDCGConfig(self.experiment, self.path_output)
                self.ndcg_result = ndcg_evaluate(ndcg_config)

                with open(os.path.join(self.path_analyses, 'ndcg_result.txt'), 'w') as writer:
                    writer.write(self.ndcg_result)

            except ZeroDivisionError:
                if self.rank == self.experiment.main_rank:
                    print(f'GPU #{self.rank}: Skipped NDCG evaluation (division by zero).')

        elif self.rank == self.experiment.main_rank:
            print(f'GPU #{self.rank}: Skipped NDCG evaluation (preconfigured for WikiData)')

        return self

    def build_dataloaders(self):
        self.train_dataloader = self._build_dataloader(self.train_data)

        if self.config.eval_ratio is not None and self.config.eval_ratio > 0:
            self.eval_dataloader = self._build_dataloader(self.eval_data)

        return self

    def _plot_loss(self, title, file_name, loss=None, d_loss=None, g_loss=None):
        if loss is not None:
            plt.plot(loss, label='Training loss')
        else:
            plt.plot(d_loss, label='Discriminator loss')
            plt.plot(g_loss, label='Generator loss')

        plt.title(title)
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig(os.path.join(self.path_analyses, file_name), dpi=300)
        plt.clf()

    @staticmethod
    def _apply_optimizer(model, config):
        optimizer = config.cls(model.parameters(), *config.args, **config.kwargs)
        return optimizer.optimizer_ if hasattr(optimizer, 'optimizer_') else optimizer

    @staticmethod
    def _apply_scheduler(optimizer, config, num_training_steps):
        scheduler = config.cls(optimizer, num_training_steps=num_training_steps,
                               *config.args, **config.kwargs)
        return scheduler.scheduler_ if hasattr(scheduler, 'scheduler_') else scheduler

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
