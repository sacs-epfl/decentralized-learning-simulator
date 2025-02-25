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
