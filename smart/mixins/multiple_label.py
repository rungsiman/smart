import torch


class MultipleLabelClassificationMixin:
    labels = ...

    class Data:
        class Tokens:
            def __init__(self, items):
                self.ids = torch.cat([item['input_ids'] for item in items], dim=0)
                self.masks = torch.cat([item['attention_mask'] for item in items], dim=0)

        def __init__(self, ids, questions, tags=None):
            self.ids = torch.tensor(ids)
            self.questions = MultipleLabelClassificationMixin.Data.Tokens(questions)

            if tags is not None:
                self.tags = torch.tensor(tags)

    def __init__(self, *args, **kwargs):
        self.name = 'multiple-label'
        self.identifier = self.resolve_identifier(kwargs.get('level', None), kwargs.get('index', None))
        super().__init__(*args, **kwargs)

    def _build_answers(self, y_ids, y_pred):
        answers = super()._get_data(y_ids)

        for answer in answers:
            answer['type'] = []

            for qid, preds in zip(y_ids, y_pred):
                if qid == answer['id'] or 'dbpedia_' + str(qid) == answer['id']:
                    answer['type'] = [self.labels[i] for i in range(len(preds)) if i < len(self.labels) and preds[i] == 1]

            if len(answer['type']) == 0:
                answer['category'] = 'resource'

        return answers

    @staticmethod
    def resolve_identifier(level=None, index=None):
        return f'level-{level}-id-{index}-multiple-label' if level is not None else 'multiple-label'
