import math
from typing import List, Tuple
from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.events import *
from dasklearn.simulation.shatter.client import ShatterClient
from dasklearn.simulation.simulation import Simulation

import networkx as nx

from dasklearn.tasks.task import Task


class ShatterSimulation(Simulation):
    CLIENT_CLASS = ShatterClient

    def __init__(self, settings: SessionSettings):
        super().__init__(settings)
        self.topologies = []
        self.finished: Dict[int, List[int, Tuple[str, int]]] = {}
        if self.settings.r == 0:
            self.settings.r = math.floor(math.log2(self.settings.participants * self.settings.k))

        self.register_event_callback(START_ROUND, "start_round")
        self.register_event_callback(FINISH_TRAIN, "finish_train")
        self.register_event_callback(SEND_CHUNKS, "send_chunks")

    def set_finished(self, round_nr: int, model: Tuple[str, int], was_online: bool = True):
        if round_nr not in self.finished:
            self.finished[round_nr] = []
        self.finished[round_nr].append(model if was_online else None)

        if len(self.finished[round_nr]) == self.settings.participants:
            self.logger.info("All clients finished round %d (t=%.3f)", round_nr, self.cur_time_in_sec())

            # Should we test?
            if self.settings.test_method == "global" and self.settings.test_interval > 0 and round_nr % self.settings.test_interval == 0:
                # Aggregate non-None models
                models = [m for m in self.finished[round_nr] if m is not None]
                task_name = Task.generate_name("agg")
                data = {"models": models, "round": round_nr, "peer": 0}
                task = Task(task_name, "aggregate", data=data)
                self.clients[0].add_compute_task(task)

                # Test the aggregated model
                test_task_name = "test_%d" % round_nr
                task = Task(test_task_name, "test", data={
                    "model": (task_name, 0), "round": round_nr,
                    "time": self.current_time, "peer": 0
                })
                self.clients[0].add_compute_task(task)

            self.finished.pop(round_nr)

    def add_topology(self):
        seed = self.settings.seed + len(self.topologies)
        g = nx.random_regular_graph(self.settings.r, self.settings.participants * self.settings.k, seed=seed)
        g = g.to_directed()
        self.topologies.append(g)
