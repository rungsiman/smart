import torch
import torch.multiprocessing as mp

from smart.data.base import Ontology, TrainingData
from smart.data.tokenizers import CustomAutoTokenizer
from smart.dist.multiprocessing import init_process
from smart.experiments.resources import Experiment
from smart.models.resources import BertForOntologyClassification
from smart.train.resources import Train
from smart.utils.devices import describe_devices
from smart.utils.reproducibility import set_seed


def process(rank, world_size, experiment, model, data, shared, lock):
    set_seed(experiment)
    torch.cuda.set_device(rank)
    init_process(rank, world_size, experiment)
    train = Train(rank, world_size, experiment, model, data, shared, lock)
    train().evaluate()


def run():
    experiment = Experiment()
    set_seed(experiment)
    describe_devices()
    print(experiment)

    world_size = experiment.num_gpu or torch.cuda.device_count()
    model = BertForOntologyClassification.from_pretrained(experiment.model)
    tokenizer = CustomAutoTokenizer(experiment)
    ontology = Ontology(experiment, tokenizer)
    data = TrainingData(experiment, ontology, tokenizer).res.clean().prepare()

    with mp.Manager() as manager:
        shared = manager.dict()
        lock = manager.Lock()
        mp.spawn(process, args=(world_size, experiment, model, data, shared, lock), nprocs=world_size, join=True)


if __name__ == '__main__':
    run()
