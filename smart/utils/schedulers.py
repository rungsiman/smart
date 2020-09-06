from transformers import get_linear_schedule_with_warmup


class LinearScheduleWithWarmup:
    def __init__(self, *args, **kwargs):
        self.scheduler_ = get_linear_schedule_with_warmup(*args, **kwargs)
