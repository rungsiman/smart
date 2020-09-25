import pandas as pd
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

from smart.mixins.paired_binary import PairedBinaryClassificationMixin
from smart.train.base import TrainBase


class TrainPairedBinaryClassification(PairedBinaryClassificationMixin, TrainBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pack(self):
        input_ids = []
        input_lids = []
        input_questions = []
        input_labels = []
        input_tags = []

        data = pd.concat([self.data.df, self.data_neg.df], ignore_index=True).to_dict(orient='records')

        for label in self.labels:
            qis_pos = [i for i in range(len(data)) if label in data[i]['type']]

            if len(qis_pos) == 0:
                neg_size = self.config.paired_binary_default_neg_size
            elif self.config.neg_size == 'mirror':
                neg_size = len(qis_pos)
            elif 'x' in self.config.neg_size:
                neg_size = len(qis_pos) * int(self.config.neg_size.replace('x', ''))
            else:
                neg_size = self.config.neg_size

            qis_neg = list(filter(lambda i: i not in qis_pos, range(len(data))))
            qis_neg = [random.choice(qis_neg) for _ in range(neg_size)]

            input_ids += [int(str(data[qi]['id']).replace('dbpedia_', '')) for qi in qis_pos] + \
                         [int(str(data[qi]['id']).replace('dbpedia_', '')) for qi in qis_neg]
            input_lids += [self.data.ontology.labels[label]['id']] * (len(qis_pos) + len(qis_neg))
            input_questions += [self.data.tokenized[data[qi]['question']] for qi in qis_pos] + \
                               [self.data_neg.tokenized[data[qi]['question']] for qi in qis_neg]
            input_labels += [self.data.ontology.labels[label]] * (len(qis_pos) + len(qis_neg))
            input_tags += [1] * len(qis_pos) + [0] * len(qis_neg)

        if len(input_ids) == 0:
            self.skipped = True
            return self

        split = train_test_split(input_ids, input_lids, input_questions, input_labels, input_tags,
                                 random_state=self.experiment.split_random_state,
                                 test_size=self.config.eval_ratio)

        train_ids, eval_ids, train_lids, eval_lids, train_questions, eval_questions, train_labels, eval_labels, train_tags, eval_tags = split

        # The tokenizer returns dictionaries containing id and mask tensors, among others.
        # These dictionaries need to be decomposed and tensors reassembled
        # before the outputs can be fed into TensorDataset
        self.train_data = TrainPairedBinaryClassification.Data(train_ids, train_lids, train_questions, train_labels, tags=train_tags)
        self.eval_data = TrainPairedBinaryClassification.Data(eval_ids, eval_lids, eval_questions, eval_labels, tags=eval_tags)
        return self

    def evaluate(self):
        if self.config.eval_ratio is None or self.config.eval_ratio == 0:
            print(f'GPU #{self.rank}: Skipped evaluation.')
            return self

        self.model.eval()

        if self.rank == self.experiment.main_rank:
            with self.lock:
                self.shared['evaluation'] = [{'y_ids': [], 'y_lids': [], 'y_true': [], 'y_prob': [], 'y_pred': []}
                                             for _ in range(self.world_size)]

        print(f'GPU #{self.rank}: Started evaluation.')
        sys.stdout.flush()
        dist.barrier()
        eval_start = time.time()

        for step, batch in (enumerate(tqdm(self.eval_dataloader, desc=f'GPU #{self.rank}: Evaluating'))
                            if self.rank == self.experiment.main_rank else enumerate(self.eval_dataloader)):
            logits = self.model(*tuple(t.cuda(self.rank) for t in batch[2:-1]), return_dict=True).logits
            probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()
            preds = (torch.sigmoid(logits) >= 0.5).long().detach().cpu().numpy().tolist()

            with self.lock:
                evaluation = self.shared['evaluation']
                evaluation[self.rank]['y_ids'] += batch[0].tolist()
                evaluation[self.rank]['y_lids'] += batch[1].tolist()
                evaluation[self.rank]['y_true'] += batch[-1].tolist()
                evaluation[self.rank]['y_prob'] += probs
                evaluation[self.rank]['y_pred'] += preds
                self.shared['evaluation'] = evaluation

        pred_size = len(self.shared['evaluation'][self.rank]['y_pred'])
        print(f'GPU #{self.rank}: Predictions for evaluation complete.')
        print(f'.. Prediction size: {pred_size}')
        dist.barrier()
        self.train_records['eval_time'] = TrainPairedBinaryClassification._format_time(time.time() - eval_start)

        if self.rank == self.experiment.main_rank:
            y_ids, y_lids, y_true, y_prob, y_pred = [], [], [], [], []

            for i in range(self.world_size):
                y_ids += self.shared['evaluation'][i]['y_ids']
                y_lids += self.shared['evaluation'][i]['y_lids']
                y_true += self.shared['evaluation'][i]['y_true']
                y_prob += self.shared['evaluation'][i]['y_prob']
                y_pred += self.shared['evaluation'][i]['y_pred']

            self.eval_report = classification_report(y_true, y_pred, digits=4)
            self.eval_dict = classification_report(y_true, y_pred, output_dict=True)
            self.eval_truths = self._get_data(y_ids)
            self.eval_answers = self._build_answers(y_ids, y_lids, y_pred)

            print(self.eval_report)

        return self

    def _build_dataloader(self, data):
        dataset = TensorDataset(data.ids, data.lids, data.questions.ids, data.questions.masks,
                                data.labels.ids, data.labels.masks, data.tags)

        self.sampler = DistributedSampler(dataset, rank=self.rank, num_replicas=self.world_size,
                                          shuffle=True, seed=self.experiment.seed)

        return DataLoader(dataset,
                          sampler=self.sampler,
                          batch_size=self.config.batch_size,
                          drop_last=self.config.drop_last)

    def _train_forward(self, batch):
        return self.model(*tuple(t.cuda(self.rank) for t in batch[2:-1]), labels=batch[-1].cuda(self.rank), return_dict=True)
