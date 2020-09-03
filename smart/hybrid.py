import torch
import torch.multiprocessing as mp

from smart.data.base import Ontology, DBpediaTrainingData
from smart.data.tokenizers import CustomAutoTokenizer
from smart.dist.multiprocessing import init_process
from smart.experiments.hybrid import Experiment
from smart.train.hybrid import DeepHybridTrain
from smart.utils.devices import describe_devices
from smart.utils.reproducibility import set_seed


def process(rank, world_size, experiment, data, shared, lock):
    set_seed(experiment)
    torch.cuda.set_device(rank)
    init_process(rank, world_size, experiment)

    data.tokenize()
    train = DeepHybridTrain(rank, world_size, experiment, data, shared, lock)
    train()


def run():
    experiment = Experiment()
    set_seed(experiment)
    describe_devices()
    print(experiment)

    world_size = experiment.num_gpu or torch.cuda.device_count()
    tokenizer = CustomAutoTokenizer(experiment)
    ontology = Ontology(experiment, tokenizer)
    data = DBpediaTrainingData(experiment, ontology, tokenizer).clean()

    with mp.Manager() as manager:
        shared = manager.dict()
        lock = manager.Lock()
        mp.spawn(process, args=(world_size, experiment, data, shared, lock), nprocs=world_size, join=True)


if __name__ == '__main__':
    run()
