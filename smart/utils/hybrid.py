import json

from smart.data.base import Ontology


class HybridConfig:
    def __init__(self, config, trainer, labels, *args, **kwargs):
        self.config = config
        self.trainer = trainer
        self.labels = labels
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        return self.config(trainer=self.trainer, labels=self.labels, *self.args, **self.kwargs)


class HybridConfigFactory:
    def __init__(self, factory, config, trainer, input_train=None, input_ontology=None, thresholds=None, *args, **kwargs):
        self.config = config
        self.trainer = trainer
        self.levels = factory(input_train, input_ontology, thresholds)
        self.hybrid = [[] for _ in range(len(self.levels))]
        self.args = args
        self.kwargs = kwargs

    def pack(self):
        for i, level in enumerate(self.levels):
            self.hybrid[i] = [HybridConfig(self.config, self.trainer, lv_set, *self.args, **self.kwargs) for lv_set in level]

        return self

    def compile(self):
        hybrid = [[] for _ in range(len(self.hybrid))]

        for i, level in enumerate(self.hybrid):
            hybrid[i] = [config() for config in level]

        return hybrid


def class_dist_thresholds(input_train=None, input_ontology=None, thresholds=None):
    ontology = Ontology(input_ontology)
    data = json.load(open(input_train))
    levels = []

    for cls in ontology.labels.values():
        cls['count'] = 0

    for question in data:
        for label in question['type']:
            if label in ontology.labels:
                ontology.labels[label]['count'] += 1

    for lv in range(ontology.max_level):
        lv_counts = [(label, item['count']) for label, item in ontology.level(lv).items()]
        lv_counts.sort(key=lambda t: t[1], reverse=True)
        lv_sets = [[] for _ in range(len(thresholds))]

        for i, threshold in enumerate(thresholds):
            for lv_count in lv_counts:
                if lv_count[1] >= threshold and (i < 1 or lv_count[1] < thresholds[i - 1]):
                    lv_sets[i].append(lv_count[0])

        levels.append(lv_sets)

    return levels[1:]
