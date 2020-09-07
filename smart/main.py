import argparse
import torch
import torch.multiprocessing as mp

from smart.data.base import Ontology, DataForTrain, DataForTest
from smart.data.tokenizers import CustomAutoTokenizer
from smart.dist.multiprocessing import init_process
from smart.experiments.hybrid import HybridExperimentConfig
from smart.experiments.literal import LiteralExperimentConfig
from smart.pipelines.hybrid import HybridTrainPipeline
from smart.pipelines.literal import LiteralTrainPipeline
from smart.utils.devices import describe_devices
from smart.utils.reproducibility import set_seed


def process(rank, world_size, experiment, stage, pipeline, shared, lock):
    set_seed(experiment)
    torch.cuda.set_device(rank)
    init_process(rank, world_size, experiment)

    tokenizer = CustomAutoTokenizer(experiment)
    ontology = Ontology(experiment, tokenizer)

    if stage == 'train':
        data = DataForTrain(experiment, ontology, tokenizer).clean().tokenize()

        if pipeline == 'literal':
            train = LiteralTrainPipeline(rank, world_size, experiment, data, shared, lock)
        else:
            train = HybridTrainPipeline(rank, world_size, experiment, data, shared, lock)

        train()

    else:
        data = DataForTest(experiment, ontology, tokenizer).clean().blind().tokenize()
        ...


def run(stage, pipeline, dataset):
    if pipeline == 'literal':
        experiment = LiteralExperimentConfig(dataset)
    else:
        experiment = HybridExperimentConfig(dataset)

    set_seed(experiment)
    describe_devices()
    print(experiment.describe())

    world_size = experiment.num_gpu or torch.cuda.device_count()

    with mp.Manager() as manager:
        shared = manager.dict()
        lock = manager.Lock()
        mp.spawn(process, args=(world_size, experiment, stage, pipeline, shared, lock), nprocs=world_size, join=True)


def verify(stage, pipeline, dataset):
    stages = ('train', 'test')
    pipelines = ('literal', 'hybrid')
    datasets = ('dbpedia', 'wikidata')

    if stage not in stages or pipeline not in pipelines or dataset not in datasets:
        print(f'ERROR: Invalid parameters. Valid parameters are:')
        print(f'.. stage: {stages}')
        print(f'.. pipeline: {pipelines}')
        print(f'.. dataset: {datasets}')
        exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', action='store')
    parser.add_argument('pipeline', action='store')
    parser.add_argument('dataset', action='store')
    args = parser.parse_args()
    verify(args.stage, args.pipeline, args.dataset)
    run(args.stage, args.pipeline, args.dataset)
