"""
These modules are responsible for scheduling model transfers while adhering to bandwidth limitations of both the
sending and receiving party.
"""
import logging
import random
from typing import List, Any, OrderedDict

from dasklearn.simulation.events import *
from dasklearn.util import MICROSECONDS


class BWScheduler:

    def __init__(self, client) -> None:
        super().__init__()
        self.bw_limit: int = 1000000  # in bytes/s, 1 MB/s by default

        # Requests that are waiting to be scheduled
        self.outgoing_requests: OrderedDict[int, Transfer] = OrderedDict()
        self.incoming_requests: OrderedDict[int, Transfer] = OrderedDict()

        # Transfers that are currently ongoing
        self.outgoing_transfers: OrderedDict[int, Transfer] = OrderedDict()
        self.incoming_transfers: OrderedDict[int, Transfer] = OrderedDict()

        self.allocated_incoming: int = 0
        self.allocated_outgoing: int = 0

        self.client = client
        self.my_id: int = self.client.index

        self.logger = logging.getLogger(self.__class__.__name__)

        self.is_active: bool = False  # Whether we are sending or receiving something
        self.became_active: int = 0

        # Statistics
        self.total_time_transmitting: int = 0
        self.total_bytes_sent = 0
        self.total_bytes_received = 0

    def get_all_transfers(self) -> List:
        """
        Get all transfers that we know about.
        """
        return list(self.incoming_transfers.values()) + list(self.outgoing_transfers.values()) + \
               list(self.incoming_requests.values()) + list(self.outgoing_requests.values())


    def register_transfer(self, transfer, is_outgoing=False):
        if not self.incoming_transfers and not self.outgoing_transfers:
            self.is_active = True
            self.became_active = self.client.simulator.current_time

        if is_outgoing:
            self.outgoing_transfers[transfer.transfer_id] = transfer
        else:
            self.incoming_transfers[transfer.transfer_id] = transfer

    def unregister_transfer(self, transfer, is_outgoing=False):
        if is_outgoing:
            if transfer.transfer_id in self.outgoing_transfers:
                self.outgoing_transfers.pop(transfer.transfer_id)
                self.allocated_outgoing -= transfer.allocated_bw
        else:
            if transfer.transfer_id in self.incoming_transfers:
                self.incoming_transfers.pop(transfer.transfer_id)
                self.allocated_incoming -= transfer.allocated_bw

        if not self.incoming_transfers and not self.outgoing_transfers:
            self.is_active = False
            self.total_time_transmitting += (self.client.simulator.current_time - self.became_active)

    def add_transfer(self, receiver_scheduler: "BWScheduler", transfer_size: int, model: str,
                     metadata: Dict[Any, Any]) -> "Transfer":
        """
        A new transfer request arrived.
        :param transfer_size: Size of the transfer, in bytes
        """
        transfer: Transfer = Transfer(self, receiver_scheduler, transfer_size, model, metadata)
        self.outgoing_requests[transfer.transfer_id] = transfer
        # self.logger.debug("Adding transfer request %d: %s => %s to the queue", transfer.transfer_id, self.my_id,
        #                   transfer.receiver_scheduler.my_id)
        self.schedule()

        return transfer

    def schedule(self):
        """
        Try to schedule pending outgoing requests and allocate bandwidth to them.
        """
        sender_bw_left: int = self.bw_limit - self.allocated_outgoing
        if sender_bw_left == 0:
            return  # Cannot accept more pending requests

        requests_scheduled: List[Transfer] = []
        for request in list(self.outgoing_requests.values()):
            receiver_bw_left = request.receiver_scheduler.bw_limit - request.receiver_scheduler.allocated_incoming
            if receiver_bw_left == 0 or sender_bw_left == 0:
                # Add this transfer as pending request in the queue of the receiver, try again later.
                if request.transfer_id not in request.receiver_scheduler.incoming_requests:
                    self.logger.debug("Sender %s adding transfer %d as pending incoming request in the scheduler of "
                                      "receiver %s", self.my_id, request.transfer_id, request.receiver_scheduler.my_id)
                    request.receiver_scheduler.incoming_requests[request.transfer_id] = request
            else:
                bw_to_allocate = min(sender_bw_left, receiver_bw_left)
                self.schedule_request(request, bw_to_allocate)
                requests_scheduled.append(request)
                sender_bw_left = self.bw_limit - self.allocated_outgoing  # Update this as it has changed

                # Do we have outgoing bandwidth left to allocate more requests?
                if sender_bw_left == 0:
                    break  # Cannot accept more pending requests                

        for request in requests_scheduled:
            self.outgoing_requests.pop(request.transfer_id)

        if sender_bw_left > 0:
            for transfer in list(self.outgoing_transfers.values()):
                self.on_receiver_inform_about_free_bandwidth(transfer)

    def schedule_request(self, request, bw_to_allocate: int):
        """
        Schedule a particular request - we know for sure that there is bandwidth available for this transfer.
        """
        self.logger.debug("Starting transfer %d: %s => %s (allocated %d bw to this transfer, s %d/%d, r %d/%d)", request.transfer_id, self.my_id,
                          request.receiver_scheduler.my_id, bw_to_allocate, self.allocated_outgoing, self.bw_limit, request.receiver_scheduler.allocated_incoming, request.receiver_scheduler.bw_limit)
        request.allocated_bw = bw_to_allocate
        self.allocated_outgoing += bw_to_allocate
        request.receiver_scheduler.allocated_incoming += bw_to_allocate
        estimated_transfer_duration = request.transfer_size / request.allocated_bw
        request.start_time = self.client.simulator.current_time
        request.last_time_updated = self.client.simulator.current_time
        assert estimated_transfer_duration >= 0

        finish_time: int = self.client.simulator.current_time + int(estimated_transfer_duration * MICROSECONDS)
        finish_transfer_event = Event(finish_time, self.client.index, FINISH_OUTGOING_TRANSFER,
                                      {"transfer": request})
        self.client.simulator.schedule(finish_transfer_event)

        self.register_transfer(request, is_outgoing=True)
        request.receiver_scheduler.register_transfer(request, is_outgoing=False)
        if request.transfer_id in request.receiver_scheduler.incoming_requests:
            request.receiver_scheduler.incoming_requests.pop(request.transfer_id)

    def on_outgoing_transfer_complete(self, transfer):
        """
        An outgoing transfer has completed.
        """
        self.logger.debug("Transfer %d: %s => %s has completed", transfer.transfer_id, self.my_id,
                          transfer.receiver_scheduler.my_id)
        transfer.finish()
        self.total_bytes_sent += transfer.transferred

        # Inform the other side
        self.unregister_transfer(transfer, is_outgoing=True)
        transfer.receiver_scheduler.on_incoming_transfer_complete(transfer)

        # Try to schedule remaining requests as we might have unallocated bandwidth at this point.
        self.schedule()

    def on_incoming_transfer_complete(self, completed_transfer):
        """
        An incoming transfer has been completed.
        We first try to allocate more bandwidth to our ongoing requests.
        Then we inform other pending incoming requests.
        """
        self.unregister_transfer(completed_transfer, is_outgoing=False)
        self.total_bytes_received += completed_transfer.transferred

        cur_time = self.client.simulator.current_time
        data = {
            "from": completed_transfer.sender_scheduler.client.index,
            "to": completed_transfer.receiver_scheduler.client.index,
            "model": completed_transfer.model,
            "metadata": completed_transfer.metadata,
        }
        incoming_model_event = Event(cur_time, self.client.index, INCOMING_MODEL, data)
        self.client.on_incoming_model(incoming_model_event)

        # Prioritize allocating bandwidth to ongoing transfers
        for transfer in list(self.incoming_transfers.values()) + list(self.incoming_requests.values()):
            # self.logger.debug("Informing sender %s about available bandwidth for transfer %d",
            #                   transfer.sender_scheduler.my_id, transfer.transfer_id)
            transfer.sender_scheduler.on_receiver_inform_about_free_bandwidth(transfer)

            incoming_bw_left: int = self.bw_limit - self.allocated_incoming
            if incoming_bw_left == 0:
                break

    def remove_transfer_finish_from_event_queue(self, transfer):
        """
        Remove an ongoing transfer completion from the event queue.
        """
        event_ind = -1
        for ind, entry in enumerate(self.client.simulator.events):
            _, _, event = entry
            if event.action == FINISH_OUTGOING_TRANSFER and event.data["transfer"] == transfer:
                event_ind = ind
                break

        assert event_ind != -1, "Ongoing transfer %d not found in event queue while removing it!" % transfer.transfer_id
        self.logger.debug("Removing completion of transfer %d from event queue", transfer.transfer_id)
        self.client.simulator.events.pop(event_ind)

    def on_receiver_inform_about_free_bandwidth(self, transfer):
        """
        A receiver of a pending transfer has informed us (the sender) about newly available bandwidth for a particular
        transfer. Adjust this transfer and try to allocate more if we can.
        """
        sender_bw_left: int = self.bw_limit - self.allocated_outgoing
        receiver_bw_left: int = transfer.receiver_scheduler.bw_limit - transfer.receiver_scheduler.allocated_incoming

        # This is either an ongoing request or a pending request
        if transfer.transfer_id in self.outgoing_transfers:
            # self.logger.debug("Sender %s got available bw notification from receiver %s for ongoing transfer %s",
            #                   self.my_id, transfer.receiver_scheduler.my_id, transfer.transfer_id)
            # It's an ongoing transfer, increase the allocated bw of this transfer accordingly
            additional_bw_to_allocate = min(sender_bw_left, receiver_bw_left)
            if additional_bw_to_allocate > 0:
                # We can allocate more bw to this transfer, do so and update everything accordingly.
                self.logger.debug("Allocating %d additional bw to transfer %d", additional_bw_to_allocate,
                                  transfer.transfer_id)

                self.remove_transfer_finish_from_event_queue(transfer)

                # First we update how much of the transfer has been completed at this point.
                transfer.update()

                # "Restart" the transfer and reschedule the completion event
                transfer.allocated_bw += additional_bw_to_allocate
                self.allocated_outgoing += additional_bw_to_allocate
                transfer.receiver_scheduler.allocated_incoming += additional_bw_to_allocate
                new_estimated_finish_time = (transfer.transfer_size - transfer.transferred) / transfer.allocated_bw
                assert new_estimated_finish_time >= 0, "Estimated finish time in the past: %f (transfer size: %d, transferred: %d)" % (new_estimated_finish_time, transfer.transfer_size, transfer.transferred)
                transfer.reschedules += 1
                finish_time: int = self.client.simulator.current_time + int(new_estimated_finish_time * MICROSECONDS)
                finish_transfer_event = Event(finish_time, self.client.index, FINISH_OUTGOING_TRANSFER,
                                              {"transfer": transfer})
                self.client.simulator.schedule(finish_transfer_event)
        elif transfer.transfer_id in self.outgoing_requests:
            # self.logger.debug("Sender %s got available bw notification from receiver %s for pending request %s",
            #                   self.my_id, transfer.receiver_scheduler.my_id, transfer.transfer_id)
            bw_to_allocate = min(sender_bw_left, receiver_bw_left)
            if bw_to_allocate > 0:
                self.schedule_request(transfer, bw_to_allocate)
                self.outgoing_requests.pop(transfer.transfer_id)
        else:
            raise RuntimeError("We do not know about request %d!" % transfer.transfer_id)
        
    def kill_transfer(self, transfer):
        self.logger.debug("Killing transfer %d: %s => %s", 
                            transfer.transfer_id, self.my_id, transfer.receiver_scheduler.my_id)
        
        # Remove from pending lists on both sender and receiver
        if transfer.transfer_id in transfer.sender_scheduler.outgoing_requests:
            transfer.sender_scheduler.outgoing_requests.pop(transfer.transfer_id)
        if transfer.transfer_id in transfer.receiver_scheduler.incoming_requests:
            transfer.receiver_scheduler.incoming_requests.pop(transfer.transfer_id)
        
        # Remove from active lists on the sender side
        if transfer.transfer_id in transfer.sender_scheduler.outgoing_transfers:
            transfer.sender_scheduler.remove_transfer_finish_from_event_queue(transfer)
            transfer.sender_scheduler.unregister_transfer(transfer, is_outgoing=True)
        
        # Remove from active lists on the receiver side
        if transfer.transfer_id in transfer.receiver_scheduler.incoming_transfers:
            transfer.receiver_scheduler.unregister_transfer(transfer, is_outgoing=False)
        
        transfer.fail()
        
        self.schedule()

    def kill_all_transfers(self):
        for transfer in list(self.incoming_transfers.values()) + list(self.outgoing_transfers.values()) + list(self.incoming_requests.values()) + list(self.outgoing_requests.values()):
            self.logger.debug("Killing transfer %d: %s => %s", transfer.transfer_id, self.my_id, transfer.receiver_scheduler.my_id)
            self.kill_transfer(transfer)


class Transfer:
    """
    Represents a bandwidth transfer.
    """

    def __init__(self, sender_scheduler: BWScheduler, receiver_scheduler: BWScheduler, transfer_size: int, model: str,
                 metadata: Dict[Any, Any]):
        self.transfer_id = random.randint(0, 100000000000)
        self.sender_scheduler: BWScheduler = sender_scheduler
        self.receiver_scheduler: BWScheduler = receiver_scheduler
        self.transfer_size: int = transfer_size
        self.model: str = model
        self.metadata: Dict[Any, Any] = metadata
        self.transferred: int = 0
        self.allocated_bw: int = 0
        self.start_time: int = -1
        self.last_time_updated: int = 0
        self.reschedules: int = 0
        self.duration: int = -1

    def finish(self):
        self.update()
        self.duration = self.receiver_scheduler.client.simulator.current_time - self.start_time

    def fail(self):
        self.update()

    def update(self):
        cur_time: int = self.sender_scheduler.client.simulator.current_time
        transferred: int = int((cur_time - self.last_time_updated) / MICROSECONDS * self.allocated_bw)
        self.transferred += transferred
        self.last_time_updated = cur_time

    def __str__(self):
        return "%s (%s => %s)" % (self.transfer_id, self.sender_scheduler.my_id, self.receiver_scheduler.my_id)
