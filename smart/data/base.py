import csv
import pandas as pd
import re


class Ontology:
    def __init__(self, experiment, tokenizer):
        self.experiment = experiment
        self.labels, self.ids = {}, []
        self.max_level = 0

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
                self.max_level = max(self.max_level, int(item[1]))

        max_len = tokenizer.find_max_len([label['text'] for label in self.labels.values()])

        for label in self.labels.values():
            encoded = tokenizer.encode(label['text'], max_len)
            label['input_ids'] = encoded['input_ids']
            label['attention_mask'] = encoded['attention_mask']

    def reverse(self, labels, level=None):
        reversed_labels = list(filter(lambda label: label not in labels, self.labels.keys()))

        if level is not None:
            reversed_labels = list(filter(lambda label: self.labels[label]['level'] == level, reversed_labels))

        return reversed_labels


class TrainingDataBase:
    def __init__(self, experiment, ontology, tokenizer, df=None, tokenized=None):
        self.experiment = experiment
        self.ontology = ontology
        self.tokenizer = tokenizer
        self.df = pd.read_json(experiment.task.input_train) if df is None else df
        self.tokenized = tokenized

    @property
    def size(self):
        return len(self.df)
    
    @property
    def resource(self):
        self.df = self.df.loc[(self.df.category == 'resource')]
        return self

    def filter(self, labels):
        removes = []

        for i, row in self.df.iterrows():
            if all(t not in labels for t in row.type):
                removes.append(i)

        self.df = self.df.drop(removes)
        return self

    def cap(self, size):
        if size is not None:
            self.df = self.df[:size]

        return self

    def tokenize(self):
        self.tokenized = {}
        max_len_questions = self.tokenizer.find_max_len(self.df.question.values)

        for question in self.df.question.values:
            self.tokenized[question] = self.tokenizer.encode(question, max_len_questions)

        return self

    def clone(self):
        ...

    def clean(self):
        self.df = self.df.drop([i for i, row in self.df.iterrows() if row.question is None])
        return self


class DBpediaTrainingData(TrainingDataBase):
    def clone(self):
        return DBpediaTrainingData(self.experiment, self.ontology, self.tokenizer, df=self.df.copy(), tokenized=self.tokenized)

    def clean(self):
        def map_labels(types):
            labels = []

            for label in types:
                if label == 'dbo:Location':
                    label = 'dbo:Place'

                labels.append(label)

            return labels

        super().clean()
        self.df = self.df.assign(type=self.df['type'].apply(map_labels))
        return self
