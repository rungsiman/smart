import numpy as np
import random
import torch


def set_seed(experiment):
    np.random.seed(experiment.seed)
    random.seed(experiment.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(experiment.seed)
    torch.cuda.manual_seed_all(experiment.seed)
    torch.manual_seed(experiment.seed)
    torch.random.manual_seed(experiment.seed)
