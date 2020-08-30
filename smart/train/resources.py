import datetime
import json
import os
import sys
import time
import torch
import torch.distributed as dist
from sklearn.metrics import classification_report
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from smart.test.evaluation import main as ndcg_evaluate


class NDCGConfig:
    def __init__(self, experiment):
        self.type_hierarchy_tsv = experiment.task.input_ontology
        self.ground_truth_json = os.path.join(experiment.task.output, 'eval_truth.json')
        self.system_output_json = os.path.join(experiment.task.output, 'eval_answers.json')


class Train:
    def __init__(self, rank, world_size, experiment, model, data, shared, lock):
        data.build_dataloaders(rank, world_size)
        num_training_steps = len(data.train_dataloader) * experiment.epochs
        model.cuda(rank)

        self.data = data
        self.rank = rank
        self.lock = lock
        self.shared = shared
        self.world_size = world_size
        self.experiment = experiment
        self.model = DistributedDataParallel(model, device_ids=[rank])
        self.optimizer = AdamW(model.parameters(),
                               lr=experiment.bert.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=experiment.bert.warmup_steps,
                                                         num_training_steps=num_training_steps)
        
        print(f'GPU #{rank}: Initialized')

    def __call__(self):
        self.train_records = {'loss': [], 'time': []}
        self.model.train()

        for epoch in range(self.experiment.epochs):
            accumulated_loss = 0
            epoch_start = time.time()
            self.data.sampler.set_epoch(epoch)

            for step, batch in (enumerate(tqdm(self.data.train_dataloader, desc=f'GPU #{self.rank}: Training'))
                                if self.rank == 0 else enumerate(self.data.train_dataloader)):
                self.optimizer.zero_grad()

                loss = self.model(*tuple(t.cuda(self.rank) for t in batch[2:]))[0]
                loss.backward()
                accumulated_loss += loss.item()

                clip_grad_norm_(parameters=self.model.parameters(),
                                max_norm=self.experiment.bert.max_grad_norm)
                
                self.optimizer.step()
                self.scheduler.step()

            average_loss = accumulated_loss / len(self.data.train_dataloader)
            training_time = Train._format_time(time.time() - epoch_start)

            if self.rank == 0:
                status = f'GPU #{self.rank}: Training epoch {epoch + 1} of {self.experiment.epochs} complete:\n'
                status += f'.. Average loss: {average_loss}\n'
                status += f'.. Training time: {training_time}'
                print(status)

            self.train_records['loss'].append(average_loss)
            self.train_records['time'].append(training_time)
        
        return self

    def evaluate(self):
        self.eval_records = {'time': []}
        self.model.eval()
        
        if self.rank == 0:
            with self.lock:
                self.shared['evaluation'] = [{'y_ids': [], 'y_lids': [], 'y_true': [], 'y_pred': []}
                                             for _ in range(self.world_size)]
        
        print(f'GPU #{self.rank}: Started evaluation')
        sys.stdout.flush()
        dist.barrier()

        for step, batch in (enumerate(tqdm(self.data.eval_dataloader, desc=f'GPU #{self.rank}: Evaluating'))
                            if self.rank == 0 else enumerate(self.data.eval_dataloader)):
            logits = self.model(*tuple(t.cuda(self.rank) for t in batch[2:-1]))
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
            
            with self.lock:
                evaluation = self.shared['evaluation']
                evaluation[self.rank]['y_ids'] += batch[0].tolist()
                evaluation[self.rank]['y_lids'] += batch[1].tolist()
                evaluation[self.rank]['y_true'] += batch[-1].tolist()
                evaluation[self.rank]['y_pred'] += preds
                self.shared['evaluation'] = evaluation
        
        pred_size = len(self.shared['evaluation'][self.rank]['y_pred'])
        print(f'GPU #{self.rank}: Predictions for evaluation complete')
        print(f'.. Prediction size: {pred_size}')
        dist.barrier()
        
        if self.rank == 0:
            y_ids, y_lids, y_true, y_pred = [], [], [], []
            
            for i in range(self.world_size):
                y_ids += self.shared['evaluation'][i]['y_ids']
                y_lids += self.shared['evaluation'][i]['y_lids']
                y_true += self.shared['evaluation'][i]['y_true']
                y_pred += self.shared['evaluation'][i]['y_pred']
            
            report = classification_report(y_true, y_pred, digits=4)
            print(report)

            with open(os.path.join(self.experiment.task.output, 'eval_result.txt'), 'w') as writer:
                writer.write(report)

            truths = self.data.get_ground_truth(y_ids)
            answers = self.data.build_answers(y_ids, y_lids, y_pred, ignore_resource_tags=True)
            json.dump(truths, open(os.path.join(self.experiment.task.output, 'eval_truth.json'), 'w'), indent=4)
            json.dump(answers, open(os.path.join(self.experiment.task.output, 'eval_answers.json'), 'w'), indent=4)

            ndcg_config = NDCGConfig(self.experiment)
            ndcg_evaluate(ndcg_config)

        return self

    @staticmethod
    def _format_time(elapsed):
        elapsed_rounded = int(round(elapsed))
        return str(datetime.timedelta(seconds=elapsed_rounded))
