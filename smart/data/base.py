import abc
import csv
import pandas as pd
import re


class Ontology:
    def __init__(self, experiment):
        self.labels, self.ids = {}, []
        self.max_level = 0

        with open(experiment.dataset.input_ontology) as reader:
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

    @abc.abstractmethod
    def clone(self):
        ...


class DataForTrain(DataBase):
    def __init__(self, experiment, df=None, ontology=None, tokenizer=None, tokenized=None):
        super().__init__(experiment, ontology=ontology, tokenizer=tokenizer, tokenized=tokenized)
        self.df = pd.read_json(experiment.dataset.input_train) if df is None else df

    def clone(self):
        return DataForTrain(self.experiment,  df=self.df.copy(),
                            ontology=self.ontology, tokenizer=self.tokenizer, tokenized=self.tokenized)


class DataForTest(DataBase):
    def __init__(self, experiment, df=None, ontology=None, tokenizer=None, tokenized=None):
        super().__init__(experiment, ontology=ontology, tokenizer=tokenizer, tokenized=tokenized)
        self.df = pd.read_json(experiment.dataset.input_test) if df is None else df

    def clone(self):
        return DataForTest(self.experiment, df=self.df.copy(),
                           ontology=self.ontology, tokenizer=self.tokenizer, tokenized=self.tokenized)

    def blind(self):
        self.df = self.df.drop(['category', 'type'])
        return self
