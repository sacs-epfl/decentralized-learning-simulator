import os
from typing import List, Tuple
from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.conflux import NodeMembershipChange
from dasklearn.simulation.conflux.client import ConfluxClient
from dasklearn.simulation.events import *
from dasklearn.simulation.asynchronous_simulation import AsynchronousSimulation
from dasklearn.tasks.task import Task


class ConfluxSimulation(AsynchronousSimulation):
    CLIENT_CLASS = ConfluxClient

    def __init__(self, settings: SessionSettings):
        super().__init__(settings)
        settings.sample_size = min(settings.sample_size, settings.participants)
        self.finished: Dict[int, List[Tuple[str, int]]] = {}
        self.register_event_callback(START_ROUND, "start_round")

    def initialize_clients(self):
        # Make sure all nodes have the same view of the network when the experiment starts
        for client in self.clients:
            for other_client in self.clients:
                join_status: NodeMembershipChange = NodeMembershipChange.JOIN if other_client.online else NodeMembershipChange.LEAVE
                client.client_manager.add_client(other_client.index, status=join_status)

        super().initialize_clients()

    def save_measurements(self):
        super().save_measurements()

        # Write contributions
        with open(os.path.join(self.data_dir, "contributions.csv"), "w") as file:
            file.write("round,client,coverage,network_speed,compute_speed\n")
            for client in self.clients:
                for info in client.contributions_in_reconstructed_models:
                    file.write("%d,%d,%f,%d,%d\n" % info)

        # Write contributions per reconstructed model
        with open(os.path.join(self.data_dir, "contributions_per_reconstructed_model.csv"), "w") as file:
            file.write("round,client,num_clients_in_model\n")
            for client in self.clients:
                for round_nr, num_contributions in client.contributions_per_model.items():
                    file.write("%d,%d,%d\n" % (round_nr, client.index, num_contributions))

        # Write round durations
        with open(os.path.join(self.data_dir, "round_durations.csv"), "w") as file:
            file.write("round,client,duration\n")
            for client in self.clients:
                for round_nr, duration in client.round_durations.items():
                    file.write("%d,%d,%f\n" % (round_nr, client.index, duration))

    def set_finished(self, round_nr: int, model: Tuple[str, int]):
        if round_nr not in self.finished:
            self.finished[round_nr] = []
        self.finished[round_nr].append(model)

        # If at least half of the current round has finished, we can probably safely proceed with wrapping up the previous round
        if len(self.finished[round_nr]) == self.settings.sample_size // 2:  # Give some slack since not all nodes might complete the round
            prev_round_nr: int = round_nr - 1
            if prev_round_nr not in self.finished:
                return

            self.logger.info("All clients in the sample finished round %d (t=%.3f)", prev_round_nr, self.cur_time_in_sec())

            # Should we test?
            if (self.settings.test_method == "global" and self.settings.test_interval > 0 and prev_round_nr % self.settings.test_interval == 0):
                # Aggregate non-None models
                task_name = Task.generate_name("agg")
                data = {"models": self.finished[prev_round_nr], "round": prev_round_nr, "peer": 0}
                task = Task(task_name, "aggregate", data=data)
                self.clients[0].add_compute_task(task)

                # Test the aggregated model
                test_task_name = "test_%d" % prev_round_nr
                task = Task(test_task_name, "test", data={
                    "model": (task_name, 0), "round": prev_round_nr,
                    "time": self.current_time, "peer": 0
                })
                self.clients[0].add_compute_task(task)

            self.finished.pop(prev_round_nr)
