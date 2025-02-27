import logging
from typing import Dict, Optional, List, Tuple

from dasklearn.simulation.conflux import NodeMembershipChange


NO_ACTIVITY_INFO = -1


class ClientManager:
    """
    The ClientManager keeps track of the population of clients that are participating in the training process.
    """

    def __init__(self, my_index: int, inactivity_threshold: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.my_index: int = my_index
        self.inactivity_threshold = inactivity_threshold
        self.last_active: Dict[int, Tuple[int, Tuple[int, NodeMembershipChange]]] = {}

    def add_client(self, other_index: int, round_active: Optional[int] = NO_ACTIVITY_INFO, status: NodeMembershipChange = NodeMembershipChange.JOIN) -> None:
        """
        Add a new client to this manager.
        """
        if other_index in self.last_active:
            return

        self.logger.debug("Client %s adding client %s to local view", self.my_index, other_index)
        self.last_active[other_index] = (round_active, (0, status))

    def remove_client(self, other_index: int) -> None:
        """
        Remove this client from this manager.
        """
        self.last_active.pop(other_index, None)

    def get_active_clients(self, round: Optional[int] = None) -> List[int]:
        active_clients = [client_id for client_id, status in self.last_active.items() if (status[1][1] != NodeMembershipChange.LEAVE or client_id == self.my_index)]
        if round:
            active_clients = [client_id for client_id in active_clients if (self.last_active[client_id][0] >= (round - self.inactivity_threshold) or client_id == self.my_index)]
        return active_clients

    def get_clients(self) -> List[int]:
        return [client_id for client_id, _ in self.last_active.items()]

    def get_num_clients(self, round: Optional[int] = None) -> int:
        """
        Return the number of clients in the local view.
        """
        return len(self.get_active_clients(round))

    def update_client_activity(self, index: int, round_active: int) -> None:
        """
        Update the status of a particular client.
        """
        info = self.last_active[index]
        self.last_active[index] = (max(self.last_active[index][0], round_active), info[1])

    def get_highest_round_in_population_view(self) -> int:
        """
        Return the highest round in the population view.
        """
        if not self.last_active:
            return -1
        return max([round for round, _ in self.last_active.values()])

    def merge_population_views(self, other_view: Dict[int, Tuple[int, Tuple[int, NodeMembershipChange]]]) -> None:
        """
        Reconcile the differences between two population views.
        """
        for client_id, info in other_view.items():
            # Is this a new joining client?
            if client_id not in self.last_active:
                # This seems to be a new client joining
                self.logger.info("Client %d adds client %d to local view", self.my_index, client_id)
                self.last_active[client_id] = other_view[client_id]
                continue

            # This client is already in the view - take its latest information

            # Check the last round activity
            last_round_active = info[0]
            if last_round_active > self.last_active[client_id][0]:
                self.update_client_activity(client_id, last_round_active)

            # Update node membership status
            membership_index = info[1][0]
            if membership_index > self.last_active[client_id][1][0]:
                self.logger.debug("Client %d updating membership status of participant %d to: %s",
                                  self.my_index, client_id, str(info[1]))
                self.last_active[client_id] = (self.last_active[client_id][0], info[1])
