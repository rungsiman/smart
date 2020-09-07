import abc
import numpy as np
import random
import sys
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from smart.train.base import TrainBase


class MultipleLabelClassificationBase(object):
    __metaclass__ = abc.ABCMeta
    labels = ...

    class Data:
        class Tokens:
            def __init__(self, items):
                self.ids = torch.cat([item['input_ids'] for item in items], dim=0)
                self.masks = torch.cat([item['attention_mask'] for item in items], dim=0)

        def __init__(self, ids, questions, tags):
            self.ids = torch.tensor(ids)
            self.questions = TrainMultipleLabelClassification.Data.Tokens(questions)
            self.tags = torch.tensor(tags)

    @abc.abstractmethod
    def _get_data(self, y_ids):
        ...

    def _build_answers(self, y_ids, y_pred):
        answers = self._get_data(y_ids)

        for answer in answers:
            answer['type'] = []

            for qid, preds in zip(y_ids, y_pred):
                if qid == answer['id'] or 'dbpedia_' + str(qid) == answer['id']:
                    answer['type'] = [self.labels[i] for i in range(len(preds)) if i < len(self.labels) and preds[i] == 1]

            if len(answer['type']) == 0:
                answer['category'] = 'resource'

        return answers


class TrainMultipleLabelClassification(TrainBase, MultipleLabelClassificationBase):
    def __init__(self, rank, world_size, experiment, model, data, labels, config, shared, lock, level=None, data_neg=None):
        self.labels = labels
        self.data_neg = data_neg
        self.identifier = f'level-{level + 1}-multiple-label' if level is not None else 'multiple-label'
        super().__init__(rank, world_size, experiment, model, data, config, shared, lock)

    def pack(self):
        ids = self.data.df.id.values
        questions = self.data.df.question.values
        labels = self.data.df.type.values
        neg_qids = []

        if self.data_neg is not None:
            if self.data.size <= self.data_neg.size:
                neg_qids = random.sample(range(self.data_neg.size), self.data.size)
            else:
                while len(neg_qids) < self.data.size:
                    neg_qids += random.sample(range(self.data_neg.size), self.data_neg.size)

                neg_qids = neg_qids[:self.data.size]

        input_ids = []
        input_questions = []
        input_tags = []

        for i, qid, question, question_labels in tqdm(zip(range(len(questions)), ids, questions, labels)):
            if len(question_labels):
                input_ids.append(int(qid.replace('dbpedia_', '')) if isinstance(qid, str) else qid)
                input_questions.append(self.data.tokenized[question])
                input_tags.append([int(label in question_labels) for label in self.labels] + [0])

                if self.data_neg is not None and (self.config.neg_size is None or i < self.config.neg_size):
                    input_ids.append(neg_qids[i])
                    input_questions.append(self.data.tokenized[self.data_neg.df.iloc[neg_qids[i]]['question']])
                    input_tags.append([0] * len(self.labels) + [1])

        split = train_test_split(input_ids, input_questions, input_tags,
                                 random_state=self.experiment.split_random_state,
                                 test_size=self.config.eval_ratio)

        train_ids, eval_ids, train_questions, eval_questions, train_tags, eval_tags = split

        self.train_data = TrainMultipleLabelClassification.Data(train_ids, train_questions, train_tags)
        self.eval_data = TrainMultipleLabelClassification.Data(eval_ids, eval_questions, eval_tags)
        return self

    def evaluate(self):
        self.model.eval()

        if self.rank == 0:
            with self.lock:
                self.shared['evaluation'] = [{'y_ids': [], 'y_true': [], 'y_pred': []} for _ in range(self.world_size)]

        print(f'GPU #{self.rank}: Started evaluation')
        sys.stdout.flush()
        dist.barrier()
        eval_start = time.time()

        for step, batch in (enumerate(tqdm(self.eval_dataloader, desc=f'GPU #{self.rank}: Evaluating'))
                            if self.rank == 0 else enumerate(self.eval_dataloader)):
            logits = self.model(*tuple(t.cuda(self.rank) for t in batch[1:-1]), return_dict=True).logits
            preds = (F.softmax(logits, dim=1) >= 0.5).long().detach().cpu().numpy().tolist()

            with self.lock:
                evaluation = self.shared['evaluation']
                evaluation[self.rank]['y_ids'] += batch[0].tolist()
                evaluation[self.rank]['y_true'] += batch[-1].tolist()
                evaluation[self.rank]['y_pred'] += preds
                self.shared['evaluation'] = evaluation

        pred_size = len(self.shared['evaluation'][self.rank]['y_pred'])
        print(f'GPU #{self.rank}: Predictions for evaluation complete')
        print(f'.. Prediction size: {pred_size}')
        dist.barrier()
        self.train_records['eval_time'] = TrainMultipleLabelClassification._format_time(time.time() - eval_start)

        if self.rank == 0:
            y_ids, y_true, y_pred = [], [], []

            for i in range(self.world_size):
                y_ids += self.shared['evaluation'][i]['y_ids']
                y_true += self.shared['evaluation'][i]['y_true']
                y_pred += self.shared['evaluation'][i]['y_pred']

            report = classification_report(np.array(y_true).reshape(-1), np.array(y_pred).reshape(-1), digits=4)
            truths = self._get_data(y_ids)
            answers = self._build_answers(y_ids, y_pred)

            self._save_evaluate(report, truths, answers)
            print(report)

        return self

    def _build_dataloader(self, data):
        dataset = TensorDataset(data.ids, data.questions.ids, data.questions.masks, data.tags)
        self.sampler = DistributedSampler(dataset, rank=self.rank, num_replicas=self.world_size,
                                          shuffle=True, seed=self.experiment.seed)

        return DataLoader(dataset,
                          sampler=self.sampler,
                          batch_size=self.config.batch_size,
                          drop_last=self.config.drop_last)

    def _train_forward(self, batch):
        return self.model(*tuple(t.cuda(self.rank) for t in batch[1:-1]), labels=batch[-1].cuda(self.rank), return_dict=True).loss
