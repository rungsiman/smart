import json
import numpy as np
import os
import random
import sys
import torch
import torch.distributed as dist
import torch.nn.functional as F
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from smart.test.evaluation import main as ndcg_evaluate

from smart.train.base import TrainBase, NDCGConfig


class TrainPairedBinaryClassification(TrainBase):
    class Data:
        class Tokens:
            def __init__(self, items):
                self.ids = torch.cat([item['input_ids'] for item in items], dim=0)
                self.masks = torch.cat([item['attention_mask'] for item in items], dim=0)

        def __init__(self, ids, lids, questions, labels, tags):
            self.ids = torch.tensor(ids)
            self.lids = torch.tensor(lids)
            self.questions = TrainPairedBinaryClassification.Data.Tokens(questions)
            self.labels = TrainPairedBinaryClassification.Data.Tokens(labels)
            self.tags = torch.tensor(tags)

    def __init__(self, rank, world_size, level, experiment, model, data, config, shared, lock):
        self.identifier = f'level-{level}-paired-binary'
        super().__init__(rank, world_size, level, experiment, model, data, config, shared, lock)

    def pack(self):
        ids = self.data.df.id.values
        questions = self.data.df.question.values
        labels = self.data.df.type.values

        input_ids = []
        input_lids = []
        input_questions = []
        input_labels = []
        input_tags = []

        # For each question, generate pairs of question-label for every label,
        # as well as for a certain amount of invalid labels (negative examples)
        for qid, question, labels_pos in tqdm(zip(ids, questions, labels)):
            labels_pos = [label for label in labels_pos if label in self.data.ontology.labels]
            question_ids = self.data.tokenized[question]
            input_questions += [question_ids] * (len(labels_pos) + self.config.neg_size)
            input_labels += [self.data.ontology.labels[label] for label in labels_pos]

            choices = list(filter(lambda label: label not in labels_pos, self.data.ontology.labels.keys()))
            labels_neg = [random.choice(choices) for _ in range(self.config.neg_size)]
            input_labels += [self.data.ontology.labels[label] for label in labels_neg]

            # Set tags to 1 for valid question-label pairs and 0 for invalid pairs
            input_tags += [1] * len(labels_pos) + [0] * len(labels_neg)
            input_ids += [int(qid.replace('dbpedia_', ''))] * (len(labels_pos) + len(labels_neg))
            input_lids += [self.data.ontology.labels[label]['id'] for label in labels_pos] + \
                          [self.data.ontology.labels[label]['id'] for label in labels_neg]

        split = train_test_split(input_ids, input_lids, input_questions, input_labels, input_tags,
                                 random_state=self.experiment.split_random_state,
                                 test_size=self.config.eval_ratio)

        train_ids, eval_ids, train_lids, eval_lids, train_questions, eval_questions, train_labels, eval_labels, train_tags, eval_tags = split

        # The tokenizer returns dictionaries containing id and mask tensors, among others.
        # These dictionaries need to be decomposed and tensors reassembled
        # before the outputs can be fed into TensorDataset
        self.train_data = TrainPairedBinaryClassification.Data(train_ids, train_lids, train_questions, train_labels, train_tags)
        self.eval_data = TrainPairedBinaryClassification.Data(eval_ids, eval_lids, eval_questions, eval_labels, eval_tags)
        return self

    def evaluate(self):
        self.model.eval()

        if self.rank == 0:
            with self.lock:
                self.shared['evaluation'] = [{'y_ids': [], 'y_lids': [], 'y_true': [], 'y_pred': []} for _ in range(self.world_size)]

        print(f'GPU #{self.rank}: Started evaluation')
        sys.stdout.flush()
        dist.barrier()

        for step, batch in (enumerate(tqdm(self.eval_dataloader, desc=f'GPU #{self.rank}: Evaluating'))
                            if self.rank == 0 else enumerate(self.eval_dataloader)):
            logits = self.model(*tuple(t.cuda(self.rank) for t in batch[2:-1]), return_dict=True).logits
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()

            with self.lock:
                evaluation = self.shared['evaluation']
                evaluation[self.rank]['y_ids'] += batch[0].tolist()
                evaluation[self.rank]['y_lids'] += batch[1].tolist()
                evaluation[self.rank]['y_true'] += batch[-1].tolist()
                evaluation[self.rank]['y_pred'] += preds
                self.shared['evaluation'] = evaluation

        pred_size = len(self.shared['evaluation'][self.rank]['y_pred'])
        print(f'GPU #{self.rank}: Predictions for evaluation complete')
        print(f'.. Prediction size: {pred_size}')
        dist.barrier()

        if self.rank == 0:
            y_ids, y_lids, y_true, y_pred = [], [], [], []

            for i in range(self.world_size):
                y_ids += self.shared['evaluation'][i]['y_ids']
                y_lids += self.shared['evaluation'][i]['y_lids']
                y_true += self.shared['evaluation'][i]['y_true']
                y_pred += self.shared['evaluation'][i]['y_pred']

            report = classification_report(y_true, y_pred, digits=4)
            print(report)

            with open(os.path.join(self.path_analyses, 'eval_result.txt'), 'w') as writer:
                writer.write(report)

            truths = self._get_data(y_ids)
            answers = self._build_answers(y_ids, y_lids, y_pred)
            json.dump(truths, open(os.path.join(self.path_output, 'eval_truth.json'), 'w'), indent=4)
            json.dump(answers, open(os.path.join(self.path_output, 'eval_answers.json'), 'w'), indent=4)

            ndcg_config = NDCGConfig(self.experiment, self.path_output)
            ndcg_evaluate(ndcg_config)

        return self

    def _train_forward(self, batch):
        return self.model(*tuple(t.cuda(self.rank) for t in batch[2:-1]), labels=batch[-1].cuda(self.rank), return_dict=True).loss

    def _build_answers(self, y_ids, y_lids, y_pred):
        answers = self._get_data(y_ids)

        for answer in answers:
            types = []

            for qid, lid, pred in zip(y_ids, y_lids, y_pred):
                if (str(qid) == answer['id'] or 'dbpedia_' + str(qid) == answer['id']) and pred == 1:
                    types.append(self.data.ontology.ids[lid])

        return answers

    def _build_dataloader(self, data):
        dataset = TensorDataset(data.ids,
                                data.lids,
                                data.questions.ids,
                                data.questions.masks,
                                data.labels.ids,
                                data.labels.masks,
                                data.tags)

        self.sampler = DistributedSampler(dataset, rank=self.rank, num_replicas=self.world_size,
                                          shuffle=True, seed=self.experiment.seed)

        return DataLoader(dataset,
                          sampler=self.sampler,
                          batch_size=self.config.batch_size,
                          drop_last=self.config.drop_last)
