import pandas as pd
import sys
import time
import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from smart.mixins.multiple_label import MultipleLabelClassificationMixin
from smart.test.base import TestBase


class TestMultipleLabelClassification(MultipleLabelClassificationMixin, TestBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self):
        self.model.eval()

        if self.rank == self.experiment.main_rank:
            with self.lock:
                self.shared['inference'] = [{'y_ids': [], 'y_pred': []} for _ in range(self.world_size)]

        print(f'GPU #{self.rank}: Started testing')
        sys.stdout.flush()
        dist.barrier()
        test_start = time.time()

        for step, batch in (enumerate(tqdm(self.test_dataloader, desc=f'GPU #{self.rank}: Testing'))
                            if self.rank == self.experiment.main_rank else enumerate(self.test_dataloader)):
            logits = self.model(*tuple(t.cuda(self.rank) for t in batch[1:]), return_dict=True).logits
            preds = (torch.sigmoid(logits) >= 0.5).long().detach().cpu().numpy().tolist()

            with self.lock:
                inference = self.shared['inference']
                inference[self.rank]['y_ids'] += batch[0].tolist()
                inference[self.rank]['y_pred'] += preds
                self.shared['inference'] = inference

        self.pred_size = len(self.shared['inference'][self.rank]['y_pred'])
        print(f'GPU #{self.rank}: Predictions for testing complete')
        print(f'.. Prediction size: {self.pred_size}')

        dist.barrier()
        self.test_records['test_time'] = TestMultipleLabelClassification._format_time(time.time() - test_start)

        if self.rank == self.experiment.main_rank:
            y_ids, y_pred = [], []

            for i in range(self.world_size):
                y_ids += self.shared['inference'][i]['y_ids']
                y_pred += self.shared['inference'][i]['y_pred']

            answers = self._build_answers(y_ids, y_pred)
            self.shared['answers'] = answers
            print(f'GPU #{self.rank}: Combined all prediction subsets')
            print(f'.. Combined prediction size: {len(y_ids)}')

        dist.barrier()
        self.answers = pd.DataFrame(self.shared['answers'])
        return self

    def pack(self):
        ids = self.data.df.id.values
        questions = self.data.df.question.values

        test_ids = []
        test_questions = []

        for qid, question in tqdm(zip(ids, questions)):
            test_ids.append(int(qid.replace('dbpedia_', '')) if isinstance(qid, str) else qid)
            test_questions.append(self.data.tokenized[question])

        self.test_data = TestMultipleLabelClassification.Data(test_ids, test_questions)
        return self

    def _build_dataloader(self, data):
        dataset = TensorDataset(data.ids, data.questions.ids, data.questions.masks)
        self.sampler = DistributedSampler(dataset, rank=self.rank, num_replicas=self.world_size,
                                          shuffle=False, seed=self.experiment.seed)
        return DataLoader(dataset,
                          sampler=self.sampler,
                          batch_size=self.config.batch_size)
