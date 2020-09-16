import datetime
import time


class TimeMonitor:
    def __init__(self):
        self.start = time.time()

    def watch(self):
        return TimeMonitor._format_time(time.time() - self.start)

    @staticmethod
    def _format_time(elapsed):
        elapsed_rounded = int(round(elapsed))
        return str(datetime.timedelta(seconds=elapsed_rounded))
