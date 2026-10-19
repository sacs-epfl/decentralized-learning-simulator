import time

from dasklearn.functions import *
from dasklearn.util.logging import setup_logging
from dasklearn.tasks.operations import execute_operation

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
            item = self.shared_queue.get()  # Blocks until an item is available
            task_name, func_name, data = item[:3]
            contract = item[3] if len(item) > 3 else None
            try:
                self.logger.debug("Worker %d executing task %s", self.index, task_name)
                res = execute_operation(func_name, self.settings, data, contract, copy_inputs=True)
                finish_time = time.time()
                self.result_queue.put((task_name, res, {"received": received_time, "finished": finish_time, "worker": self.index}))
            except Exception as exc:
                self.logger.error("=== Worker %d failed to execute task %s ===", self.index, task_name)
                self.logger.exception(exc)
                self.result_queue.put(("error", task_name, None))
                break
