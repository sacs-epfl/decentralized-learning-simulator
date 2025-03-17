import os
from dasklearn.session_settings import SessionSettings
from dasklearn.simulation.conflux import NodeMembershipChange
from dasklearn.simulation.conflux.client import ConfluxClient
from dasklearn.simulation.events import *
from dasklearn.simulation.asynchronous_simulation import AsynchronousSimulation


class ConfluxSimulation(AsynchronousSimulation):
    CLIENT_CLASS = ConfluxClient

    def __init__(self, settings: SessionSettings):
        super().__init__(settings)
        settings.sample_size = min(settings.sample_size, settings.participants)
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
