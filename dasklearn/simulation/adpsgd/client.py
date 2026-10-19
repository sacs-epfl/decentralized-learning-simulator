"""Active/passive asynchronous pairwise SGD.

Passive workers keep computing single gradients while exchanges proceed. A
finished gradient is applied to the current model, which may have been mixed
since computation started. Active workers train a block, exchange, then repeat.
"""
from dasklearn.simulation.asynchronous_client import AsynchronousClient
from dasklearn.simulation.events import Event, START_TRAIN, TEST
from dasklearn.tasks.task import Task
from dasklearn.tasks.references import TaskRef


class ADPSGDClient(AsynchronousClient):
    def __init__(self, simulator, index: int):
        super().__init__(simulator, index)
        self.active = True

    def init_client(self, event: Event):
        # Both roles need a real initial model before either training or mixing.
        initial = self.create_compute_task("initialize", {"peer": self.index, "round": 0, "time": event.time}, name=Task.generate_name("initialize"))
        self.own_model = TaskRef(initial.name, 0)
        self.publish_model(0, self.own_model)
        self.simulator.schedule(Event(event.time, self.index, START_TRAIN,
                                      {"model": self.own_model, "round": 0}))
        if self.simulator.settings.test_method != "snapshot" and self.simulator.settings.test_period > 0:
            self.simulator.schedule(Event(event.time + self.simulator.settings.test_period, self.index, TEST))

    def get_train_time(self):
        block_time = super().get_train_time()
        return block_time if self.active else max(1, block_time // self.simulator.settings.learning.local_steps)

    def finish_train(self, event: Event):
        self.age += 1
        if self.active:
            self.own_model = event.data["model"]
            self.publish_model(self.age, self.own_model)
            self.disseminate_model()
        else:
            # Do not overwrite a mix that arrived during gradient computation.
            task = self.create_compute_task("gradient_update", {
                "model": self.own_model, "gradient_model": event.data["model"],
                "round": self.age, "time": event.time, "peer": self.index}, name=Task.generate_name("gradient_update"))
            self.own_model = TaskRef(task.name, 0)
            self.publish_model(self.age, self.own_model)
            self.simulator.schedule(Event(event.time, self.index, START_TRAIN,
                                          {"model": self.own_model, "round": self.age}))

    def process_incoming_model(self, event: Event):
        if not self.active:
            # Return the exact pre-mix value, so both sides use the same pair.
            self.send(event.data["from"], self.own_model, size_bytes=self.simulator.model_size, metadata={"age": self.age}, overhead_bytes=0)
        self.aggregate([(event.data["from"], event.data["model"], event.data["metadata"]["age"])])
        self.publish_model(self.age, self.own_model)
        if self.active:
            self.simulator.schedule(Event(event.time, self.index, START_TRAIN,
                                          {"model": self.own_model, "round": self.age}))
