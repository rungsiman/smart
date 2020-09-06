import sys
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from smart.test.base import TestBase
from smart.train.multiple_label import MultipleLabelClassificationBase


class TestMultipleLabelClassification(TestBase, MultipleLabelClassificationBase):
    class Data:
        class Tokens:
            def __init__(self, items):
                self.ids = torch.cat([item['input_ids'] for item in items], dim=0)
                self.masks = torch.cat([item['attention_mask'] for item in items], dim=0)

        def __init__(self, ids, questions):
            self.ids = torch.tensor(ids)
            self.questions = TestMultipleLabelClassification.Data.Tokens(questions)

    def __init__(self, rank, world_size, experiment, model, data, labels, config, shared, lock, level=None):
        self.labels = labels
        self.identifier = f'level-{level + 1}-multiple-label' if level is not None else 'multiple-label'
        super().__init__(rank, world_size, experiment, model, data, config, shared, lock)

    def __call__(self):
        self.model.eval()

        if self.rank == 0:
            with self.lock:
                self.shared['inference'] = [{'y_ids': [], 'y_pred': []} for _ in range(self.world_size)]

        print(f'GPU #{self.rank}: Started testing')
        sys.stdout.flush()
        dist.barrier()
        test_start = time.time()

        for step, batch in (enumerate(tqdm(self.test_dataloader, desc=f'GPU #{self.rank}: Testing'))
                            if self.rank == 0 else enumerate(self.test_dataloader)):
            logits = self.model(*tuple(t.cuda(self.rank) for t in batch[1:]), return_dict=True).logits
            preds = (F.softmax(logits, dim=1) >= 0.5).long().detach().cpu().numpy().tolist()

            with self.lock:
                inference = self.shared['inference']
                inference[self.rank]['y_ids'] += batch[0].tolist()
                inference[self.rank]['y_pred'] += preds
                self.shared['inference'] = inference

        pred_size = len(self.shared['inference'][self.rank]['y_pred'])
        print(f'GPU #{self.rank}: Predictions for testing complete')
        print(f'.. Prediction size: {pred_size}')
        dist.barrier()
        self.test_records['test_time'] = TestMultipleLabelClassification._format_time(time.time() - test_start)

        if self.rank == 0:
            y_ids, y_pred = [], []

            for i in range(self.world_size):
                y_ids += self.shared['inference'][i]['y_ids']
                y_pred += self.shared['inference'][i]['y_pred']

            answers = self._build_answers(y_ids, y_pred)
            self._save_evaluate(answers)

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
        return DataLoader(dataset, batch_size=self.config.batch_size, drop_last=self.config.drop_last)
