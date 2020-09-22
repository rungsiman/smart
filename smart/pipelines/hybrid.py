import json
import os

from smart.data.base import Ontology
from smart.data.tokenizers import CustomAutoTokenizer
from smart.pipelines.base import PipelineBase
from smart.utils.configs import override
from smart.utils.monitoring import TimeMonitor


class HybridTrainPipeline(PipelineBase):
    def __call__(self):
        self.all_data_size = self.data.size
        self.data = self.data.resource
        stopwatch = TimeMonitor()
        pipeline_records, pipeline_eval = [], []
        ontology = Ontology(self.experiment.dataset.input_ontology)

        for level in range(1, ontology.max_level + 1):
            processed_labels = []

            if level <= len(self.experiment.dataset.hybrid):
                for index, config in enumerate(self.experiment.dataset.hybrid[level - 1]):
                    processed_labels += config.labels

                    if self.experiment.dataset.selective_train is None or \
                            f'id-{index}' in self.experiment.dataset.selective_train or \
                            f'level-{level}' in self.experiment.dataset.selective_train or \
                            f'level-{level}-id-{index}' in self.experiment.dataset.selective_train:
                        self._process(level, index, config, config.labels, ontology,
                                      pipeline_records, pipeline_eval, set_num_labels=True)

            if self.experiment.dataset.hybrid_default is not None:
                if self.experiment.dataset.selective_train is None or \
                        f'default' in self.experiment.dataset.selective_train or \
                        f'level-{level}' in self.experiment.dataset.selective_train or \
                        f'level-{level}-default' in self.experiment.dataset.selective_train:
                    config = self.experiment.dataset.hybrid_default
                    labels_reversed = ontology.reverse(processed_labels, level)
                    self._process(level, 'default', config, labels_reversed, ontology,
                                  pipeline_records, pipeline_eval)

            elif self.rank == self.experiment.main_rank:
                print(f'GPU #{self.rank}: Skipped training default classifier on level {level}.')

        if self.rank == self.experiment.main_rank:
            json.dump(pipeline_records, open(os.path.join(self.experiment.dataset.output_analyses, 'pipeline_train_records.json'), 'w'), indent=4)
            json.dump(pipeline_eval, open(os.path.join(self.experiment.dataset.output_analyses, 'pipeline_eval_records.json'), 'w'), indent=4)

            with open(os.path.join(self.experiment.dataset.output_analyses, 'pipeline_train_records.txt'), 'w') as writer:
                writer.write(f'Approximate training time: {stopwatch.watch()}')

    def _process(self, level, index, config, labels, ontology, pipeline_records, pipeline_eval, set_num_labels=False):
        labels_reversed = ontology.reverse(labels, level)

        tokenizer = CustomAutoTokenizer(config)
        ontology.tokenize(tokenizer)
        self.data.tokenize(ontology, tokenizer)

        bert_config = config.bert_config.from_pretrained(config.model)
        override(bert_config, config)

        if set_num_labels:
            bert_config.num_labels = len(labels) + 1

        # For reversed data, if filter only with (labels, reverse=True), the rest of the data on all levels will still be included.
        # If filter only with (labels_reversed), questions having both positive and negative types will be included.
        # Use both to filter only negative samples on the same level.
        data_hybrid = self.data.clone().filter(labels)
        data_reversed = self.data.clone().filter(labels, reverse=True)

        if len(labels) and data_hybrid.size > 0:
            data_hybrid.cap(config.data_size_cap)
            model = config.classifier.from_pretrained(config.model, config=bert_config)
            train = config.trainer(self.rank, self.world_size, self.experiment, model, data_hybrid, labels,
                                   config, self.shared, self.lock, level=level, index=index, data_neg=data_reversed)

            status = f'GPU #{self.rank}: Training #{index} "{train.name}" on level {level}.\n'
            status += f'.. Type count: {len(labels)}\n'
            status += f'.. Data size: {data_hybrid.size} of {self.data.size} ({self.all_data_size} including literal)\n'
            status += f'.. Negative data size: {data_reversed.size}'

            if config.data_size_cap is not None:
                status += f' (Data cap applied)'

            if self.rank == self.experiment.main_rank:
                print(status)

            if not train.skipped:
                train().evaluate().save()

                pipeline_records.append({'level': level, 'index': index, 'classification': train.name,
                                         'data_size': data_hybrid.size, 'data_neg_size': data_reversed.size,
                                         'label_size':  len(labels), 'reversed_label_size': len(labels_reversed)})

                if train.eval_dict is not None:
                    pipeline_eval.append({'level': level, 'index': index, 'classification': train.name,
                                          'f1-micro': train.eval_dict.get('micro avg', None),
                                          'f1-macro': train.eval_dict.get('macro avg', None),
                                          'f1-weighted': train.eval_dict.get('weighted avg', None)})

            elif self.rank == self.experiment.main_rank:
                print(f'GPU #{self.rank}: Skipped training #{index} on level {level} (no eligible data)')

        elif self.rank == self.experiment.main_rank:
            print(f'GPU #{self.rank}: Skipped training #{index} on level {level}')


class HybridTestPipeline(PipelineBase):
    def __call__(self):
        stopwatch = TimeMonitor()
        pipeline_records = []
        ontology = Ontology(self.experiment.dataset.input_ontology)

        for level in range(1, ontology.max_level + 1):
            processed_labels = []

            if level <= len(self.experiment.dataset.hybrid):
                for index, config in enumerate(self.experiment.dataset.hybrid[level - 1]):
                    self._process(level, index, config, config.labels, pipeline_records, processed_labels)
                    processed_labels += config.labels

            if self.experiment.dataset.hybrid_default is not None:
                config = self.experiment.dataset.hybrid_default
                reversed_labels = ontology.reverse(processed_labels, level)
                self._process(level, 'default', config, reversed_labels, pipeline_records, processed_labels)

            elif self.rank == self.experiment.main_rank:
                print(f'GPU #{self.rank}: Skipped testing default classifier on level {level}.')

        if self.rank == self.experiment.main_rank:
            self.data.apply_test_strategy()
            self.data.save(os.path.join(self.experiment.paths.final, f'answers-{self.experiment.dataset.name}.json'))
            json.dump(pipeline_records, open(os.path.join(self.experiment.dataset.output_analyses, 'pipeline_test_records.json'), 'w'), indent=4)

            num_q_with_answers = self.data.count_questions_with_answers()
            status = f'.. Approximate testing time: {stopwatch.watch()}\n'
            status += f'.. All questions: {self.data.size}\n'
            status += f'.. Questions with answers: {num_q_with_answers} (%.4f%%)\n' % (num_q_with_answers / self.data.size * 100)
            status += f'.. Unique answers: {self.data.count_answers()}'

            with open(os.path.join(self.experiment.dataset.output_analyses, 'pipeline_test_records.txt'), 'w') as writer:
                writer.write(status.replace('.. ', ''))

            print(f'GPU #{self.rank}: Testing complete.\n' + status)

    def _process(self, level, index, config, labels, pipeline_records, processed_labels):
        identifier = config.tester.resolve_identifier(level, index)
        path_models = os.path.join(self.experiment.dataset.output_models, identifier)

        if not os.path.exists(path_models):
            print(f'GPU #{self.rank}: Skipped testing #{index} on level {level} (trained model not found)')
            return

        bert_config = config.bert_config.from_pretrained(path_models)
        override(bert_config, config)

        tokenizer = CustomAutoTokenizer(config, path_models)
        ontology = Ontology(self.experiment.dataset.input_ontology).tokenize(tokenizer)
        self.data.tokenize(ontology, tokenizer)

        # Assuming that the default index will always be the paired-label classification
        if level == 1 or (self.experiment.dataset.test_strategy != 'dependent' and index != 'default'):
            if index != 'default':
                data_test = self.data.clone().resource
            else:
                data_test = self.data.clone().resource.filter(processed_labels, reverse=True)

        else:
            data_test = self.data.clone().resource.filter(ontology.parents(labels))

        if data_test.size > 0:
            model = config.classifier.from_pretrained(config.model, config=bert_config)
            test = config.tester(self.rank, self.world_size, self.experiment, model, data_test, labels, config,
                                 self.shared, self.lock, level=level, index=index)

            status = f'GPU #{self.rank}: Testing #{index} "{test.name}" on level {level}.\n'
            status += f'.. Type count: {len(labels)}\n'
            status += f'.. Data size: {data_test.size} of {self.data.size}'

            if config.data_size_cap is not None:
                status += f' (Data cap applied)'

            print(status)

            if not test.skipped:
                test()
                test.data.blind().assign_answers(test.answers)
                test.data.assign_missing_answers()
                test.save()

                num_existing_answers = self.data.count_answers()
                self.data.assign_answers(test.answers)
                self.data.assign_missing_answers()

                if self.rank == self.experiment.main_rank:
                    status = f'GPU #{self.rank}: Testing #{index} "{test.name}" on level {level} complete.\n'
                    status += f'.. Data size: {self.data.size} ({test.data.size} processed by this classifier)\n'
                    status += f'.. Answer count: {self.data.count_answers() - num_existing_answers}\n'
                    status += f'.. Accumulated answer count: {self.data.count_answers()}'
                    print(status)

                pipeline_records.append({'level': level, 'index': index, 'classification': test.name,
                                         'answer_count': self.data.count_answers() - num_existing_answers,
                                         'accumulated_answer_count': self.data.count_answers(),
                                         'data_processed_size': test.data.size,
                                         'data_size': self.data.size})

            elif self.rank == self.experiment.main_rank:
                print(f'GPU #{self.rank}: Skipped testing #{index} on level {level} (no eligible data)')

        elif self.rank == self.experiment.main_rank:
            print(f'GPU #{self.rank}: Skipped testing #{index} on level {level} (no data to test after filtering)')
