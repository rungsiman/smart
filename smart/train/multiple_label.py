import numpy as np
import random
import sys
import time
import torch
import torch.distributed as dist
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from smart.mixins.multiple_label import MultipleLabelClassificationMixin
from smart.train.base import TrainBase


class TrainMultipleLabelClassification(MultipleLabelClassificationMixin, TrainBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pack(self):
        ids = self.data.df.id.values
        questions = self.data.df.question.values
        labels = self.data.df.type.values

        neg_ids = self.data_neg.df.id.values
        neg_questions = self.data_neg.df.question.values

        neg_orders = []

        if self.config.neg_size == 'mirror':
            neg_size = self.data.size
        elif 'x' in self.config.neg_size:
            neg_size = self.data.size * int(self.config.neg_size.replace('x', ''))
        else:
            neg_size = self.config.neg_size

        if self.data_neg is not None:
            if neg_size <= self.data_neg.size:
                neg_orders = random.sample(range(self.data_neg.size), neg_size)
            else:
                while len(neg_orders) < neg_size:
                    neg_orders += random.sample(range(self.data_neg.size), self.data_neg.size)

                neg_orders = neg_orders[:neg_size]

        input_ids = []
        input_questions = []
        input_tags = []

        for i, qid, question, question_labels in tqdm(zip(range(len(questions)), ids, questions, labels)):
            if len(question_labels):
                input_ids.append(int(qid.replace('dbpedia_', '')) if isinstance(qid, str) else qid)
                input_questions.append(self.data.tokenized[question])
                input_tags.append([int(label in question_labels) for label in self.labels] + [0])

            else:
                print(f'WARNING: [TrainMultiLabel.pack] No ground-truth labels for question: {qid}')

        for i in neg_orders:
            input_ids.append(int(neg_ids[i].replace('dbpedia_', '')) if isinstance(neg_ids[i], str) else neg_ids[i])
            input_questions.append(self.data_neg.tokenized[neg_questions[i]])
            input_tags.append([0] * len(self.labels) + [0])

        if self.config.eval_ratio is None or self.config.eval_ratio == 0:
            self.train_data = TrainMultipleLabelClassification.Data(input_ids, input_questions, tags=input_tags)
        else:
            split = train_test_split(input_ids, input_questions, input_tags,
                                     random_state=self.experiment.split_random_state,
                                     test_size=self.config.eval_ratio)

            train_ids, eval_ids, train_questions, eval_questions, train_tags, eval_tags = split

            self.train_data = TrainMultipleLabelClassification.Data(train_ids, train_questions, tags=train_tags)
            self.eval_data = TrainMultipleLabelClassification.Data(eval_ids, eval_questions, tags=eval_tags)

        return self

    def evaluate(self):
        if self.config.eval_ratio is None or self.config.eval_ratio == 0:
            print(f'GPU #{self.rank}: Skipped evaluation.')
            return self

        self.model.eval()

        if self.rank == self.experiment.main_rank:
            with self.lock:
                self.shared['evaluation'] = [{'y_ids': [], 'y_true': [], 'y_pred': []} for _ in range(self.world_size)]

        print(f'GPU #{self.rank}: Started evaluation.')
        sys.stdout.flush()
        dist.barrier()
        eval_start = time.time()

        for step, batch in (enumerate(tqdm(self.eval_dataloader, desc=f'GPU #{self.rank}: Evaluating'))
                            if self.rank == self.experiment.main_rank else enumerate(self.eval_dataloader)):
            logits = self.model(*tuple(t.cuda(self.rank) for t in batch[1:-1]), return_dict=True).logits
            preds = (torch.sigmoid(logits) >= 0.5).long().detach().cpu().numpy().tolist()

            with self.lock:
                evaluation = self.shared['evaluation']
                evaluation[self.rank]['y_ids'] += batch[0].tolist()
                evaluation[self.rank]['y_true'] += batch[-1].tolist()
                evaluation[self.rank]['y_pred'] += preds
                self.shared['evaluation'] = evaluation

        pred_size = len(self.shared['evaluation'][self.rank]['y_pred'])
        print(f'GPU #{self.rank}: Predictions for evaluation complete.')
        print(f'.. Prediction size: {pred_size}')
        dist.barrier()
        self.train_records['eval_time'] = TrainMultipleLabelClassification._format_time(time.time() - eval_start)

        if self.rank == self.experiment.main_rank:
            y_ids, y_true, y_pred = [], [], []

            for i in range(self.world_size):
                y_ids += self.shared['evaluation'][i]['y_ids']
                y_true += self.shared['evaluation'][i]['y_true']
                y_pred += self.shared['evaluation'][i]['y_pred']

            self.eval_report = classification_report(np.array(y_true).reshape(-1), np.array(y_pred).reshape(-1), digits=4)
            self.eval_dict = classification_report(np.array(y_true).reshape(-1), np.array(y_pred).reshape(-1), output_dict=True)
            self.eval_truths = self._get_data(y_ids)
            self.eval_answers = self._build_answers(y_ids, y_pred)

            print(self.eval_report)

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
        return self.model(*tuple(t.cuda(self.rank) for t in batch[1:-1]), labels=batch[-1].cuda(self.rank), return_dict=True)
