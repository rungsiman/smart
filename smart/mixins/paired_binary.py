import torch


class PairedBinaryClassificationMixin:
    data = ...

    class Data:
        class Tokens:
            def __init__(self, items):
                self.ids = torch.cat([item['input_ids'] for item in items], dim=0)
                self.masks = torch.cat([item['attention_mask'] for item in items], dim=0)

        def __init__(self, ids, lids, questions, labels, tags=None):
            self.ids = torch.tensor(ids)
            self.lids = torch.tensor(lids)
            self.questions = PairedBinaryClassificationMixin.Data.Tokens(questions)
            self.labels = PairedBinaryClassificationMixin.Data.Tokens(labels)

            if tags is not None:
                self.tags = torch.tensor(tags)

    def __init__(self, *args, **kwargs):
        self.name = 'paired-binary'
        self.identifier = self.resolve_identifier(kwargs.get('level', None), kwargs.get('index', None))
        super().__init__(*args, **kwargs)

    def _build_answers(self, y_ids, y_lids, y_pred, y_prob=None):
        answers = super()._get_data(y_ids)

        if y_prob is None:
            for answer in answers:
                answer['type'] = []

                for qid, lid, pred in zip(y_ids, y_lids, y_pred):
                    if (qid == answer['id'] or 'dbpedia_' + str(qid) == answer['id']) and pred[0] == 1:
                        answer['type'].append(self.data.ontology.ids[lid])

                if len(answer['type']) == 0:
                    answer['category'] = 'resource'

        else:
            for answer in answers:
                answer['type'] = []
                prob_threshold = 0.5

                for qid, lid, prob in zip(y_ids, y_lids, y_prob):
                    if qid == answer['id'] or 'dbpedia_' + str(qid) == answer['id'] and prob[0] >= prob_threshold:
                        answer['type'] = [self.data.ontology.ids[lid]]
                        prob_threshold = prob[0]

                if len(answer['type']) == 0:
                    answer['category'] = 'resource'

        return answers

    @staticmethod
    def resolve_identifier(level=None, index=None):
        return f'level-{level}-id-{index}-paired-binary' if level is not None else 'paired-binary'
