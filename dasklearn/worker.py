import time

from dasklearn.functions import *
from dasklearn.util.logging import setup_logging

torch.multiprocessing.set_sharing_strategy('file_system')


class Worker:

    def __init__(self, shared_queue, result_queue, index: int, settings: SessionSettings):
        self.shared_queue = shared_queue
        self.result_queue = result_queue
        self.index: int = index
        self.settings = settings
        setup_logging(self.settings.data_dir, "worker_%d.log" % self.index, log_level=settings.log_level)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info("Worker %d initialized", self.index)

    def start(self):
        while True:
            received_time = time.time()
            task_name, func_name, data = self.shared_queue.get()  # Blocks until an item is available
            try:
                self.logger.debug("Worker %d executing task %s", self.index, task_name)
                if func_name not in globals():
                    raise RuntimeError("Task function %s not found!" % func_name)
                f = globals()[func_name]

                res = f(self.settings, data)
                finish_time = time.time()
                self.result_queue.put((task_name, res, {"received": received_time, "finished": finish_time, "worker": self.index}))
            except Exception as exc:
                self.logger.exception(exc)
                self.result_queue.put(("error", None))
                break
