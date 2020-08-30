import csv
import pandas as pd
import random
import re
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


class Ontology:
    def __init__(self, experiment, tokenizer):
        self.experiment = experiment
        self.labels, self.ids = {}, []

        with open(experiment.task.input_ontology) as reader:
            items = csv.reader(reader, delimiter='\t')

            # Skip header row
            next(items)

            for i, item in enumerate(items):
                text = ' '.join(re.findall('[A-Z][^A-Z]*', item[0])).lower()
                self.labels[item[0]] = {'id': i - 1,
                                        'text': text,
                                        'level': int(item[1]),
                                        'parent': item[2]}
                self.ids.append(item[0])

        max_len = tokenizer.find_max_len([label['text'] for label in self.labels.values()])

        for label in self.labels.values():
            label['ids'] = tokenizer.encode(label['text'], max_len)


class TrainingData:
    class Pack:
        class Components:
            def __init__(self, items):
                self.ids = torch.cat([item['input_ids'] for item in items], dim=0)
                self.masks = torch.cat([item['attention_mask'] for item in items], dim=0)

        def __init__(self, ids, lids, texts, labels, tags):
            self.ids = torch.tensor(ids)
            self.lids = torch.tensor(lids)
            self.texts = TrainingData.Pack.Components(texts)
            self.labels = TrainingData.Pack.Components(labels)
            self.tags = torch.tensor(tags)

    def __init__(self, experiment, ontology, tokenizer):
        self.experiment = experiment
        self.ontology = ontology
        self.tokenizer = tokenizer

        self.df = pd.read_json(experiment.task.input_train)

        if experiment.data_size is not None:
            self.df = self.df[:experiment.data_size]
    
    @property
    def res(self):
        self.df = self.df.loc[(self.df['category'] == 'resource')]
        return self

    def clean(self):
        def map_labels(uris):
            labels = {}

            for uri in uris:
                if uri == 'dbo:Location':
                    uri = 'dbo:Place'

                labels[uri] = self.ontology.labels[uri]

            return labels

        self.df = self.df.assign(text=self.df['question'].map(lambda text: 'None' if text is None else text.lower()))
        self.df = self.df.assign(labels=self.df['type'].apply(map_labels))

        return self

    def prepare(self):
        ids = self.df.id.values
        texts = self.df.text.values
        labels = self.df.labels.values

        max_len_texts = self.tokenizer.find_max_len(texts)

        input_ids = []
        input_lids = []
        input_texts = []
        input_labels = []
        input_tags = []

        # For each question, generate pairs of question-label for every label,
        # as well as for a certain amount of invalid labels (negative examples)
        for qid, text, labels_pos in tqdm(zip(ids, texts, labels)):
            text_ids = self.tokenizer.encode(text, max_len_texts)
            input_texts += [text_ids] * (len(labels_pos) + self.experiment.neg_size)
            input_labels += [label['ids'] for label in labels_pos.values()]

            choices = list(filter(lambda label: label not in labels_pos.keys(), self.ontology.labels.keys()))
            labels_neg = [random.choice(choices) for _ in range(self.experiment.neg_size)]
            input_labels += [self.ontology.labels[label]['ids'] for label in labels_neg]

            # Set tags to 1 for valid question-label pairs and 0 for invalid pairs
            input_tags += [1] * len(labels_pos) + [0] * len(labels_neg)
            input_ids += [int(qid.replace('dbpedia_', ''))] * (len(labels_pos) + len(labels_neg))
            input_lids += [label['id'] for label in labels_pos.values()] + [self.ontology.labels[label]['id'] for label in labels_neg]

        split = train_test_split(input_ids, input_lids, input_texts, input_labels, input_tags,
                                 random_state=self.experiment.split_random_state, 
                                 test_size=self.experiment.test_ratio)
        
        train_ids, eval_ids, train_lids, eval_lids, train_texts, eval_texts, train_labels, eval_labels, train_tags, eval_tags = split

        # The tokenizer returns dictionaries containing id and mask tensors, among others.
        # These dictionaries need to be decomposed and tensors reassembled
        # before the outputs can be fed into TensorDataset
        self.train_data = TrainingData.Pack(train_ids, train_lids, train_texts, train_labels, train_tags)
        self.eval_data = TrainingData.Pack(eval_ids, eval_lids, eval_texts, eval_labels, eval_tags)

        return self

    def build_dataloaders(self, rank, world_size):
        self.train_dataloader = self._build_dataloader(self.train_data, rank, world_size)
        self.eval_dataloader = self._build_dataloader(self.eval_data, rank, world_size)

        return self

    def _build_dataloader(self, data, rank, world_size):
        dataset = TensorDataset(data.ids,
                                data.lids,
                                data.texts.ids,
                                data.texts.masks,
                                data.labels.ids,
                                data.labels.masks,
                                data.tags)
        
        self.sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, 
                                          shuffle=True, seed=self.experiment.seed)

        return DataLoader(dataset,
                          sampler=self.sampler,
                          batch_size=self.experiment.batch_size,
                          drop_last=self.experiment.drop_last)

    def get_ground_truth(self, y_ids):
        y_dbpedia_ids = ['dbpedia_' + str(qid) for qid in y_ids]
        answers = self.df.loc[self.df['id'].isin(y_ids)].to_dict('records')
        answers += self.df.loc[self.df['id'].isin(y_dbpedia_ids)].to_dict('records')

        for answer in answers:
            del answer['text']
            del answer['labels']

        return answers

    def build_answers(self, y_ids, y_lids, y_pred, ignore_resource_tags=False):
        answers = self.get_ground_truth(y_ids)

        for answer in answers:
            types = list(filter(lambda t: t in ('boolean', 'date', 'string', 'number'), answer['type'])) if ignore_resource_tags else answer['type']

            for qid, lid, pred in zip(y_ids, y_lids, y_pred):
                if (str(qid) == answer['id'] or 'dbpedia_' + str(qid) == answer['id']) and pred == 1:
                    types.append(self.ontology.ids[lid])

        return answers
