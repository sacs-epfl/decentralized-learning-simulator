import logging
import random

from dasklearn.simulation.events import *
from dasklearn.util import MICROSECONDS

class SlotTransfer:
    """
    Represents a transfer that uses a fixed slot-based bandwidth.
    """
    def __init__(self, sender, receiver, transfer_size, model, metadata, slot_index=None):
        self.transfer_id = random.randint(0, 100000000000)
        self.sender = sender            # Sender's SlotBWScheduler instance
        self.receiver = receiver        # Receiver's SlotBWScheduler instance
        self.transfer_size = transfer_size
        self.model = model
        self.metadata = metadata
        self.transferred = 0
        self.slot_index = slot_index    # The index of the slot allocated at the sender side
        self.start_time = -1
        self.last_time_updated = 0
        # Each slot gets an equal share of the sender's total bandwidth.
        self.allocated_bw = sender.slot_bw

    def update(self):
        """
        Update the amount of data transferred based on the time elapsed.
        """
        current_time = self.sender.client.simulator.current_time
        time_elapsed = (current_time - self.last_time_updated) / MICROSECONDS
        self.transferred += int(self.allocated_bw * time_elapsed)
        self.last_time_updated = current_time

    def is_complete(self):
        """
        Check if the transfer is complete.
        """
        return self.transferred >= self.transfer_size

    def finish(self):
        """
        Finalize the transfer.
        """
        self.update()


class SlotBWScheduler:

    def __init__(self, client, num_slots=5, total_bw=1000000):
        self.client = client
        self.num_slots = num_slots
        self.bw_limit = total_bw
        self.slot_bw = total_bw // num_slots  # Each slot gets an equal fraction of the total bandwidth.
        
        self.outgoing_slots = [None] * num_slots
        self.incoming_slots = [None] * num_slots

        self.total_bytes_sent: int = 0
        self.total_bytes_received: int = 0
        
        self.logger = logging.getLogger(self.__class__.__name__)

    def _find_free_slot(self, slot_list):
        """Return the index of a free slot, or None if all slots are occupied."""
        for index, slot in enumerate(slot_list):
            if slot is None:
                return index
        return None

    def _find_transfer_slot(self, slot_list, transfer):
        """Return the slot index that holds the given transfer, or None if not found."""
        for index, slot in enumerate(slot_list):
            if slot == transfer:
                return index
        return None
    
    def has_free_incoming_slot(self):
        return self._find_free_slot(self.incoming_slots) is not None
    
    def has_free_outgoing_slot(self):
        return self._find_free_slot(self.outgoing_slots) is not None

    def add_transfer(self, receiver_scheduler, transfer_size, model, metadata):
        """
        Initiate a new transfer from this scheduler (sender) to the receiver's scheduler.
        The transfer is scheduled immediately if both sender and receiver have a free slot;
        otherwise it is queued until a slot becomes available.
        """
        transfer = SlotTransfer(self, receiver_scheduler, transfer_size, model, metadata)
        # Attempt to find a free outgoing slot at the sender.
        sender_slot_index = self._find_free_slot(self.outgoing_slots)
        if sender_slot_index is None:
            raise Exception("No free slot available at the sender.")
        
        receiver_slot_index = receiver_scheduler._find_free_slot(receiver_scheduler.incoming_slots)
        if receiver_slot_index is None:
            raise Exception("No free slot available at the receiver.")

        transfer.slot_index = sender_slot_index
        self.outgoing_slots[sender_slot_index] = transfer
        receiver_scheduler.incoming_slots[receiver_slot_index] = transfer
        transfer.start_time = self.client.simulator.current_time
        transfer.last_time_updated = transfer.start_time

        # Calculate the estimated duration based on the fixed slot bandwidth.
        estimated_duration = transfer_size / min(self.slot_bw, receiver_scheduler.slot_bw)
        finish_time = self.client.simulator.current_time + int(estimated_duration * MICROSECONDS)
        finish_event = Event(finish_time, self.client.index, FINISH_OUTGOING_TRANSFER, {"transfer": transfer})
        self.client.simulator.schedule(finish_event)
        self.logger.debug(f"Transfer {transfer.transfer_id} scheduled in slot {sender_slot_index} "
                            f"with bandwidth {self.slot_bw} bytes/s and duration {estimated_duration} s.")
        return transfer
    
    def on_outgoing_transfer_complete(self, transfer):
        transfer.finish()
        self.total_bytes_sent += transfer.transferred

        self.outgoing_slots[transfer.slot_index] = None

        # Inform the other side
        transfer.receiver.on_incoming_transfer_complete(transfer)

    def on_incoming_transfer_complete(self, completed_transfer):
        """
        An incoming transfer has been completed.
        """
        receiver_slot_idx: int = self._find_transfer_slot(self.incoming_slots, completed_transfer)
        self.incoming_slots[receiver_slot_idx] = None
        self.total_bytes_received += completed_transfer.transferred

        cur_time = self.client.simulator.current_time
        data = {
            "from": completed_transfer.sender.client.index,
            "to": completed_transfer.sender.client.index,
            "model": completed_transfer.model,
            "metadata": completed_transfer.metadata,
        }
        incoming_model_event = Event(cur_time, self.client.index, INCOMING_MODEL, data)
        self.client.on_incoming_model(incoming_model_event)

    def kill_transfer(self, transfer):
        """
        Kill a transfer in progress.
        """
        in_idx = self._find_transfer_slot(self.incoming_slots, transfer)
        out_idx = self._find_transfer_slot(self.outgoing_slots, transfer)
        if in_idx is not None:
            self.incoming_slots[in_idx] = None
        if out_idx is not None:
            self.outgoing_slots[out_idx] = None
        transfer.finish()
        self.remove_transfer_finish_from_event_queue(transfer)

        # Remove the transfer from the other side as well
        other_schedule = transfer.sender if transfer.sender != self else transfer.receiver
        in_idx = other_schedule._find_transfer_slot(other_schedule.incoming_slots, transfer)
        out_idx = other_schedule._find_transfer_slot(other_schedule.outgoing_slots, transfer)
        if in_idx is not None:
            other_schedule.incoming_slots[in_idx] = None
        if out_idx is not None:
            other_schedule.outgoing_slots[out_idx] = None
        #other_schedule.remove_transfer_finish_from_event_queue(transfer)

        self.logger.debug(f"Transfer {transfer.transfer_id} killed.")

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
