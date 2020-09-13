import json
import os

from smart.data.base import Ontology
from smart.data.tokenizers import CustomAutoTokenizer
from smart.pipelines.base import PipelineBase
from smart.utils.configs import override


class HybridTrainPipeline(PipelineBase):
    def __call__(self):
        self.data = self.data.resource
        pipeline_records, pipeline_eval = [], []
        ontology = Ontology(self.experiment.dataset.input_ontology)

        for level in range(1, ontology.max_level + 1):
            processed_labels = []

            if level <= len(self.experiment.dataset.hybrid):
                for index, config in enumerate(self.experiment.dataset.hybrid[level - 1]):
                    self._process(level, index, config, config.labels, ontology,
                                  processed_labels, pipeline_records, pipeline_eval, set_num_labels=True)

            config = self.experiment.dataset.hybrid_default
            labels_reversed = ontology.reverse(processed_labels, level)
            self._process(level, 'default', config, labels_reversed, ontology,
                          processed_labels, pipeline_eval, pipeline_records)

        if self.rank == 0:
            json.dump(pipeline_records, open(os.path.join(self.experiment.dataset.output_analyses, 'pipeline_train_records.json'), 'w'), indent=4)
            json.dump(pipeline_eval, open(os.path.join(self.experiment.dataset.output_analyses, 'pipeline_eval_records.json'), 'w'), indent=4)

    def _process(self, level, index, config, labels, ontology, processed_labels, pipeline_records, pipeline_eval, set_num_labels=False):
        labels_reversed = ontology.reverse(labels, level)
        processed_labels += labels

        tokenizer = CustomAutoTokenizer(config)
        ontology.tokenize(tokenizer)
        self.data.tokenize(ontology, tokenizer)

        bert_config = config.bert_config.from_pretrained(config.model)
        override(bert_config, config)

        if set_num_labels:
            bert_config.num_labels = len(labels) + 1

        # For reversed data, if filter only with (labels, reverse=True), the rest of the data on all levels will still be included.
        # If filter only with (labels_reversed), questions having both primary and secondary types will be included.
        data_hybrid = self.data.clone().filter(labels)
        data_reversed = self.data.clone().filter(labels, reverse=True).filter(labels_reversed)

        if len(labels) and data_hybrid.size > 0:
            data_hybrid.cap(config.data_size_cap)
            model = config.classifier.from_pretrained(config.model, config=bert_config)
            train = config.trainer(self.rank, self.world_size, self.experiment, model, data_hybrid, labels,
                                   config, self.shared, self.lock, level=level, data_neg=data_reversed)

            status = f'GPU #{self.rank}: Training #{index} "{train.name}" on level {level}.\n'
            status += f'.. Type count: {len(labels)}\n'
            status += f'.. Data size: {data_hybrid.size} of {self.data.size}\n'
            status += f'.. Negative data size: {data_reversed.size}'

            if config.data_size_cap is not None:
                status += f' (Data cap applied)'

            if self.rank == 0:
                print(status)

            train().evaluate().save()

            pipeline_records.append({'level': level, 'index': index, 'classification': train.name,
                                     'data_size': data_hybrid.size, 'data_neg_size': data_reversed.size,
                                     'label_size':  len(labels), 'reversed_label_size': len(labels_reversed)})

            if train.eval_dict is not None:
                pipeline_eval.append({'level': level, 'index': index, 'classification': train.name,
                                      'f1-micro': train.eval_dict.get('micro avg', None),
                                      'f1-macro': train.eval_dict.get('macro avg', None),
                                      'f1-weighted': train.eval_dict.get('weighted avg', None)})

        elif self.rank == 0:
            print(f'GPU #{self.rank}: Skipped training #{index} on level {level}')


class HybridTestPipeline(PipelineBase):
    def __call__(self):
        pipeline_records = []
        ontology = Ontology(self.experiment.dataset.input_ontology)

        for level in range(1, ontology.max_level + 1):
            if level <= len(self.experiment.dataset.hybrid):
                processed_labels = []

                for index, config in enumerate(self.experiment.dataset.hybrid[level - 1]):
                    self._process(level, index, config, config.labels, ontology, pipeline_records, processed_labels)

                config = self.experiment.dataset.hybrid_default
                labels_reversed = ontology.reverse(processed_labels, level)
                self._process(level, 'default', config, labels_reversed, ontology, pipeline_records, processed_labels)

        if self.rank == 0:
            self.data.save(os.path.join(self.experiment.paths.final, 'answers.json'))
            json.dump(pipeline_records, open(os.path.join(self.experiment.dataset.output_analyses, 'pipeline_test_records.json'), 'w'), indent=4)

    def _process(self, level, index, config, labels, ontology, pipeline_records, processed_labels, test_remaining_label=False):
        identifier = config.tester.resolve_identifier(level)
        labels_parents = ontology.parents(ontology.reverse(processed_labels, level) if test_remaining_label else labels)
        path_models = os.path.join(self.experiment.dataset.output_models, identifier)

        bert_config = config.bert_config.from_pretrained(path_models)
        override(bert_config, config)

        tokenizer = CustomAutoTokenizer(config, path_models)
        ontology = Ontology(self.experiment.dataset.input_ontology).tokenize(tokenizer)
        self.data.tokenize(ontology, tokenizer)

        data_test = self.data.clone() if level == 1 else self.data.clone().filter(labels_parents)

        if len(labels) and data_test.size > 0:
            model = config.classifier.from_pretrained(config.model, config=bert_config)
            test = config.tester(self.rank, self.world_size, self.experiment, model, data_test, labels, config,
                                 self.shared, self.lock, level=level)

            status = f'GPU #{self.rank}: Testing #{index} "{test.name}" on level {level}.\n'
            status += f'.. Type count: {len(labels)}\n'
            status += f'.. Data size: {data_test.size} of {self.data.size}'

            if config.data_size_cap is not None:
                status += f' (Data cap applied)'

            print(status)

            test()

            if not test.skipped:
                test.data.assign_answers(test.answers)
                test.save()

                num_existing_answers = self.data.count_answers()
                self.data.assign_answers(test.answers)

                if self.rank == 0:
                    status = f'GPU #{self.rank}: Testing #{index} "{test.name}" on level {level} complete.\n'
                    status += f'.. Answer count: {self.data.count_answers() - num_existing_answers}\n'
                    status += f'.. Accumulated answer count: {self.data.count_answers()}'
                    print(status)

                pipeline_records.append({'level': level, 'index': index, 'classification': test.name,
                                         'answer_count': self.data.count_answers() - num_existing_answers,
                                         'accumulated_answer_count': self.data.count_answers()})

            elif self.rank == 0:
                print(f'GPU #{self.rank}: Skipped testing #{index} on level {level} (No eligible data)')

        elif self.rank == 0:
            print(f'GPU #{self.rank}: Skipped testing #{index} on level {level}')
