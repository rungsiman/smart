import abc
import copy
import csv
import numpy as np
import pandas as pd
import re


class Ontology:
    def __init__(self, location):
        self.labels, self.ids = {}, []
        self.max_level = 0

        with open(location) as reader:
            items = csv.reader(reader, delimiter='\t')

            # Skip header row
            next(items)

            for i, item in enumerate(items):
                text = ' '.join(re.findall('[A-Z][^A-Z]*', item[0])).lower()
                self.labels[item[0]] = {'id': i,
                                        'text': text,
                                        'level': int(item[1]),
                                        'parent': item[2]}
                self.ids.append(item[0])
                self.max_level = max(self.max_level, int(item[1]))

    def tokenize(self, tokenizer):
        max_len = tokenizer.find_max_len([label['text'] for label in self.labels.values()])

        for label in self.labels.values():
            encoded = tokenizer.encode(label['text'], max_len)
            label['input_ids'] = encoded['input_ids']
            label['attention_mask'] = encoded['attention_mask']

        return self

    def reverse(self, labels, level=None):
        reversed_labels = list(filter(lambda label: label not in labels, self.labels.keys()))

        if level is not None:
            reversed_labels = list(filter(lambda label: self.labels[label]['level'] == level, reversed_labels))

        return reversed_labels

    def parents(self, labels):
        return [self.labels[label]['parent'] for label in labels]

    def level(self, lv):
        return {key: item for key, item in self.labels.items() if item['level'] == lv}

    def trace(self, label, reverse=False):
        branch = self._trace(label, [])[1:]

        if not reverse:
            branch.reverse()

        return branch

    def _trace(self, label, branch):
        branch.append(label)

        if self.labels[label]['level'] > 1:
            return self._trace(self.labels[label]['parent'], branch)

        return branch


class DataBase:
    __metaclass__ = abc.ABCMeta
    df = ...
    ontology = ...
    tokenizer = ...

    def __init__(self, experiment, ontology=None, tokenizer=None, tokenized=None):
        self.experiment = experiment
        self.ontology = ontology
        self.tokenizer = tokenizer
        self.tokenized = tokenized

    @property
    def size(self):
        return len(self.df)

    @property
    def literal(self):
        self.df = self.df.loc[(self.df.category != 'resource')]
        return self
    
    @property
    def resource(self):
        self.df = self.df.loc[(self.df.category == 'resource')]
        return self

    def filter(self, labels, reverse=False):
        removes = []

        if reverse:
            for i, row in self.df.iterrows():
                if any(t in labels for t in row.type):
                    removes.append(i)

        else:
            for i, row in self.df.iterrows():
                if all(t not in labels for t in row.type):
                    removes.append(i)

        self.df = self.df.drop(removes)
        return self

    def cap(self, size):
        if size is not None:
            self.df = self.df[:size]

        return self

    def tokenize(self, ontology, tokenizer):
        self.ontology = ontology
        self.tokenizer = tokenizer
        self.tokenized = {}

        max_len_questions = self.tokenizer.find_max_len(self.df.question.values)

        for question in self.df.question.values:
            self.tokenized[question] = self.tokenizer.encode(question, max_len_questions)

        return self

    def clean(self):
        def map_labels(types):
            labels = []

            for label in types:
                if label not in ('dbo:MedicalSpecialty', 'something', 'dbo:Location'):
                    labels.append(label)

            return labels

        self.df = self.df.drop([i for i, row in self.df.iterrows() if row.question is None])

        if 'type' in self.df.columns:
            self.df = self.df.assign(type=self.df['type'].apply(map_labels))

        return self

    def save(self, location):
        self.df.to_json(location, orient="records", indent=4)
        return self

    @abc.abstractmethod
    def clone(self):
        ...


class DataForTrain(DataBase):
    def __init__(self, experiment, df=None, ontology=None, tokenizer=None, tokenized=None):
        super().__init__(experiment, ontology=ontology, tokenizer=tokenizer, tokenized=tokenized)
        self.df = pd.read_json(experiment.dataset.input_train) if df is None else df

    def clone(self):
        df_dict = self.df.to_dict(orient='records')
        return DataForTest(self.experiment, df=pd.DataFrame(copy.deepcopy(df_dict)),
                           ontology=self.ontology, tokenizer=self.tokenizer, tokenized=self.tokenized)


class DataForTest(DataBase):
    def __init__(self, experiment, df=None, ontology=None, tokenizer=None, tokenized=None):
        super().__init__(experiment, ontology=ontology, tokenizer=tokenizer, tokenized=tokenized)
        self.df = pd.read_json(experiment.dataset.input_test) if df is None else df

    def clone(self):
        df_dict = self.df.to_dict(orient='records')
        return DataForTest(self.experiment, df=pd.DataFrame(copy.deepcopy(df_dict)),
                           ontology=self.ontology, tokenizer=self.tokenizer, tokenized=self.tokenized)

    def blind(self):
        self.df['category'] = ''
        self.df['type'] = [[] for _ in range(len(self.df))]
        return self

    def assign_categories(self):
        for i, row in self.df.iterrows():
            if len(row.type) == 0:
                category = 'resource'
            elif 'boolean' in row.type:
                category = 'boolean'
            else:
                category = 'literal'

            self.df.loc[self.df.id == row.id, 'category'] = category

        return self

    def assign_answers(self, df):
        df_dict = self.df.to_dict(orient='records')

        for row in df_dict:
            if row['id'] in df.id.values:
                row['type'] += df[df.id == row['id']]['type'].tolist()[0]

        self.df = pd.DataFrame(df_dict)
        return self

    def assign_missing_answers(self):
        df_dict = self.df.to_dict(orient='records')

        # Removed following SMART Task organizers' reply on getting rid of dbo:Location entirely
        # for row in df_dict:
        #    if 'dbo:Place' in row['type'] and 'dbo:Location' not in row['type']:
        #        row['type'].append('dbo:Location')

        self.df = pd.DataFrame(df_dict)
        return self

    def count_answers(self):
        return int(np.array([len(ans) for ans in self.df['type'].tolist()]).sum())

    def count_questions_with_answers(self):
        return int(np.array([int(len(ans) > 0) for ans in self.df['type'].tolist()]).sum())

    def apply_test_strategy(self):
        if self.experiment.dataset.test_strategy in ('top-down', 'bottom-up'):
            df_dict = self.df.to_dict(orient='records')

            for row in df_dict:
                if row['category'] == 'resource':
                    if self.experiment.dataset.test_strategy == 'top-down':
                        labels = []

                        for label in row['type']:
                            branch = self.ontology.trace(label)

                            if all([element in row['type'] for element in branch]):
                                labels.append(label)

                        row['type'] = labels

                    else:
                        labels = []

                        for label in row['type']:
                            labels += self.ontology.trace(label)

                        for label in labels:
                            if label not in row['type']:
                                row['type'].append(label)

            self.df = pd.DataFrame(df_dict)

        return self
