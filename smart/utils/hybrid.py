import json

from smart.data.base import Ontology


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


class HybridConfig:
    def __init__(self, config, trainer, labels):
        self.config = config
        self.trainer = trainer
        self.labels = labels

    def __call__(self, *args, **kwargs):
        return self.config(trainer=self.trainer, labels=self.labels, *args, **kwargs)


class HybridConfigFactory:
    def __init__(self, *, factory, config, trainer, **kwargs):
        self.config = config
        self.trainer = trainer
        self.levels = factory(**kwargs)
        self.hybrid = [[] for _ in range(len(self.levels))]

    def pack(self):
        for i, level in enumerate(self.levels):
            self.hybrid[i] = [HybridConfig(self.config, self.trainer, lv_set) for lv_set in level]

        return self

    def compile(self):
        hybrid = [[] for _ in range(len(self.hybrid))]

        for i, level in enumerate(self.hybrid):
            hybrid[i] = [config() for config in level]

        return hybrid
