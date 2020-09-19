import json

from smart.data.base import Ontology
from smart.utils.configs import select


class HybridConfig:
    def __init__(self, config, trainer, labels, **kwargs):
        self.config = config
        self.trainer = trainer
        self.labels = labels
        self.kwargs = kwargs

    def __call__(self, **kwargs):
        return self.config(trainer=self.trainer, labels=self.labels, **{**self.kwargs, **kwargs})


class HybridConfigFactory:
    def __init__(self, factory, config, trainer, input_train=None, input_ontology=None, thresholds=None, **kwargs):
        self.config = config
        self.trainer = trainer
        self.labels = factory(input_train, input_ontology, thresholds, **select('level'))
        self.hybrid = [[] for _ in range(len(self.labels))]
        self.kwargs = kwargs

    def pack(self):
        for lv_i, lv_label_sets in enumerate(self.labels):
            self.hybrid[lv_i] = [HybridConfig(self.config, self.trainer, lv_label_set,
                                              **select(self.kwargs, 'all',
                                                       f'id-{set_i}-all',
                                                       f'level-{lv_i + 1}-all',
                                                       f'level-{lv_i + 1}-id-{set_i}'))
                                 for set_i, lv_label_set in enumerate(lv_label_sets)]

        return self

    def compile(self):
        hybrid = [[] for _ in range(len(self.hybrid))]

        for i, level in enumerate(self.hybrid):
            hybrid[i] = [config() for config in level]

        return hybrid


def class_dist_thresholds(input_train=None, input_ontology=None, thresholds=None, **kwargs):
    ontology = Ontology(input_ontology)
    data = json.load(open(input_train))
    levels = []

    for cls in ontology.labels.values():
        cls['count'] = 0

    for question in data:
        for label in question['type']:
            if label in ontology.labels:
                ontology.labels[label]['count'] += 1

    for lv_i in range(ontology.max_level):
        lv_counts = [(label, item['count']) for label, item in ontology.level(lv_i).items()]
        lv_counts.sort(key=lambda t: t[1], reverse=True)
        lv_sets = [[] for _ in range(len(thresholds))]

        for threshold_i, threshold in enumerate(kwargs.get(f'{lv_i}-thresholds', thresholds)):
            for lv_count in lv_counts:
                if lv_count[1] >= threshold and (threshold_i < 1 or lv_count[1] < thresholds[threshold_i - 1]):
                    lv_sets[threshold_i].append(lv_count[0])

        levels.append(lv_sets)

    return levels[1:]
