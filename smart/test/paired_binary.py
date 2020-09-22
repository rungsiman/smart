import pandas as pd
import sys
import time
import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from smart.mixins.paired_binary import PairedBinaryClassificationMixin
from smart.test.base import TestBase


class TestPairedBinaryClassification(PairedBinaryClassificationMixin, TestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self):
        self.model.eval()

        if self.rank == self.experiment.main_rank:
            with self.lock:
                self.shared['inference'] = [{'y_ids': [], 'y_lids': [], 'y_pred': []} for _ in range(self.world_size)]

        print(f'GPU #{self.rank}: Started testing.')
        sys.stdout.flush()
        dist.barrier()
        test_start = time.time()

        for step, batch in (enumerate(tqdm(self.test_dataloader, desc=f'GPU #{self.rank}: Testing'))
                            if self.rank == self.experiment.main_rank else enumerate(self.test_dataloader)):
            logits = self.model(*tuple(t.cuda(self.rank) for t in batch[2:]), return_dict=True).logits
            preds = (torch.sigmoid(logits) >= 0.5).long().detach().cpu().numpy().tolist()

            with self.lock:
                inference = self.shared['inference']
                inference[self.rank]['y_ids'] += batch[0].tolist()
                inference[self.rank]['y_lids'] += batch[1].tolist()
                inference[self.rank]['y_pred'] += preds
                self.shared['inference'] = inference

        self.pred_size = len(self.shared['inference'][self.rank]['y_pred'])
        print(f'GPU #{self.rank}: Predictions for testing complete')
        print(f'.. Prediction size: {self.pred_size}')
        dist.barrier()
        self.test_records['test_time'] = TestPairedBinaryClassification._format_time(time.time() - test_start)

        if self.rank == self.experiment.main_rank:
            y_ids, y_lids, y_pred = [], [], []

            for i in range(self.world_size):
                y_ids += self.shared['inference'][i]['y_ids']
                y_lids += self.shared['inference'][i]['y_lids']
                y_pred += self.shared['inference'][i]['y_pred']

            answers = self._build_answers(y_ids, y_lids, y_pred)
            self.shared['answers'] = answers

        dist.barrier()
        self.answers = pd.DataFrame(self.shared['answers'])
        return self

    def pack(self, **kwargs):
        ids = self.data.df.id.values
        questions = self.data.df.question.values
        answers = self.data.df.type.values

        test_ids = []
        test_lids = []
        test_questions = []
        test_labels = []

        for qid, question, types in tqdm(zip(ids, questions, answers)):
            question_ids = self.data.tokenized[question]

            if self.level == 1:
                labels = [label for label in self.labels if self.data.ontology.labels[label]['count'] >= self.config.test_classes_min_dist]
            else:
                labels = [label for label in self.labels if self.data.ontology.labels[label]['parent'] in types]

            test_ids += [int(qid.replace('dbpedia_', '')) if isinstance(qid, str) else qid] * len(labels)
            test_lids += [self.data.ontology.labels[label]['id'] for label in labels]
            test_questions += [question_ids] * len(labels)
            test_labels += [self.data.ontology.labels[label] for label in labels]

        if len(test_ids) == 0:
            self.skipped = True
            return self

        self.test_data = TestPairedBinaryClassification.Data(test_ids, test_lids, test_questions, test_labels)

        return self

    def _build_dataloader(self, data):
        dataset = TensorDataset(data.ids, data.lids, data.questions.ids, data.questions.masks, data.labels.ids, data.labels.masks)
        self.sampler = DistributedSampler(dataset, rank=self.rank, num_replicas=self.world_size,
                                          shuffle=True, seed=self.experiment.seed)
        return DataLoader(dataset,
                          sampler=self.sampler,
                          batch_size=self.config.batch_size)
