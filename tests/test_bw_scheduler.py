from typing import List, Tuple
import pytest
import logging

from dasklearn.simulation.bandwidth_scheduler import BWScheduler, Transfer

# Override MICROSECONDS for testing simplicity.
MICROSECONDS = 1

# Dummy event action constants (normally imported from dasklearn.simulation.events)
FINISH_OUTGOING_TRANSFER = "finish_outgoing_transfer"
INCOMING_MODEL = "incoming_model"

# Dummy Event class for testing purposes
class DummyEvent:
    def __init__(self, time, client_index, action, data):
        self.time = time
        self.client_index = client_index
        self.action = action
        self.data = data

# Dummy Simulator that tracks current_time and scheduled events
class DummySimulator:
    def __init__(self):
        self.current_time = 1000  # starting time in microseconds
        self.events: List[Tuple[int, int, DummyEvent]] = []

    def schedule(self, event):
        self.events.append((event.time, 0, event))

# Dummy Client that provides an index, simulator, and an on_incoming_model callback
class DummyClient:
    def __init__(self, index):
        self.index = index
        self.simulator = DummySimulator()
        self.incoming_model_events = []

    def on_incoming_model(self, event):
        self.incoming_model_events.append(event)

# It is assumed that BWScheduler and Transfer are defined in the same module or imported accordingly.
# from your_module import BWScheduler, Transfer

# -------------------------------
# Pytest fixtures for setting up tests
# -------------------------------

@pytest.fixture
def setup_schedulers():
    client1 = DummyClient(index=1)
    client2 = DummyClient(index=2)
    scheduler1 = BWScheduler(client1)
    scheduler2 = BWScheduler(client2)
    return scheduler1, scheduler2, client1, client2


def test_register_and_unregister_transfer(setup_schedulers):
    scheduler1, scheduler2, client1, client2 = setup_schedulers
    # Initially, the scheduler should not be active.
    assert not scheduler1.is_active
    transfer = Transfer(scheduler1, scheduler2, 1000, "dummy_model", {})

    # Register the transfer (simulate starting a transfer)
    scheduler1.register_transfer(transfer, is_outgoing=True)
    assert scheduler1.is_active
    assert scheduler1.became_active == client1.simulator.current_time

    # Simulate passage of time.
    client1.simulator.current_time += 5000
    scheduler1.unregister_transfer(transfer, is_outgoing=True)
    assert not scheduler1.is_active
    # The total time transmitting should be increased by the time difference.
    assert scheduler1.total_time_transmitting == 5000


def test_schedule_request(setup_schedulers):
    scheduler1, scheduler2, client1, client2 = setup_schedulers
    transfer_size = 100000  # in bytes
    model = "test_model"
    metadata = {"key": "value"}

    # Set a known current time.
    client1.simulator.current_time = 1000

    transfer = scheduler1.add_transfer(scheduler2, transfer_size, model, metadata)
    # The allocated bandwidth should be the minimum of sender and receiver limits.
    expected_bw = min(scheduler1.bw_limit, scheduler2.bw_limit)
    assert transfer.allocated_bw == expected_bw
    assert transfer.start_time == 1000
    # The transfer should now be registered as an ongoing transfer.
    assert transfer in scheduler1.outgoing_transfers
    assert transfer in scheduler2.incoming_transfers
    # Check that a finish event has been scheduled.
    assert any(event.data.get("transfer") == transfer for _, _, event in client1.simulator.events)


def test_add_transfer(setup_schedulers):
    scheduler1, scheduler2, _, _ = setup_schedulers
    scheduler1.bw_limit = 200000
    scheduler2.bw_limit = 300000
    
    # Add a transfer
    transfer = scheduler1.add_transfer(scheduler2, 100000, "test_model", {})
    assert transfer.allocated_bw == scheduler1.bw_limit
    assert scheduler1.get_allocated_outgoing_bw() == scheduler1.bw_limit
    assert scheduler2.get_allocated_incoming_bw() == scheduler1.bw_limit
    assert transfer in scheduler1.outgoing_transfers
    assert transfer in scheduler2.incoming_transfers

    # Adding another transfer should not schedule it
    transfer2 = scheduler1.add_transfer(scheduler2, 100000, "test_model", {})
    assert transfer2.allocated_bw == 0
    assert scheduler1.get_allocated_outgoing_bw() == scheduler1.bw_limit
    assert scheduler2.get_allocated_incoming_bw() == scheduler1.bw_limit
    assert transfer2 in scheduler1.outgoing_requests


def test_on_outgoing_transfer_complete(setup_schedulers):
    scheduler1, scheduler2, client1, client2 = setup_schedulers
    transfer = Transfer(scheduler1, scheduler2, 100000, "test_model", {})
    transfer.allocated_bw = 500000
    client1.simulator.current_time = 1000
    transfer.start_time = 1000
    transfer.last_time_updated = 1000
    scheduler1.outgoing_transfers.append(transfer)
    scheduler2.incoming_transfers.append(transfer)

    # Advance time to simulate data transfer progress.
    client1.simulator.current_time += 5000

    # Complete the transfer.
    scheduler1.on_outgoing_transfer_complete(transfer)

    # finish() should have updated the transferred bytes.
    assert transfer.get_transferred_bytes() >= 0
    # The transfer should be unregistered from the sender.
    assert transfer not in scheduler1.outgoing_transfers
    # The sender's total_bytes_sent should have been updated.
    assert scheduler1.total_bytes_sent == transfer.get_transferred_bytes()
    # The receiver should have unregistered the transfer as well.
    assert transfer not in scheduler2.incoming_transfers


def test_on_receiver_inform_about_free_bandwidth(setup_schedulers):
    scheduler1, scheduler2, client1, _ = setup_schedulers
    transfer = Transfer(scheduler1, scheduler2, 100000, "test_model", {})
    transfer.allocated_bw = 300000
    scheduler1.outgoing_transfers.append(transfer)
    scheduler2.incoming_transfers.append(transfer)

    # Manually schedule an initial finish event for this transfer.
    finish_event = DummyEvent(
        time=client1.simulator.current_time + 1000,
        client_index=client1.index,
        action=FINISH_OUTGOING_TRANSFER,
        data={"transfer": transfer}
    )
    client1.simulator.events.append((finish_event.time, 0, finish_event))

    original_bw = transfer.allocated_bw

    # Simulate that available bandwidth is present.
    scheduler1.on_receiver_inform_about_free_bandwidth(transfer)
    # The transfer should have received additional bandwidth.
    assert transfer.allocated_bw > original_bw
    # Check that a new finish event for this transfer is scheduled.
    assert any(event.data.get("transfer") == transfer for (_, _, event) in client1.simulator.events)


def test_kill_transfer_active(setup_schedulers):
    scheduler1, scheduler2, client1, client2 = setup_schedulers
    transfer = scheduler1.add_transfer(scheduler2, 1000, "test_model", {})
    assert transfer in scheduler1.outgoing_transfers
    assert transfer in scheduler2.incoming_transfers
    assert len(client1.simulator.events) == 1

    scheduler1.kill_transfer(transfer)

    # Verify that the transfer has been removed from active lists on both sides.
    assert transfer not in scheduler1.outgoing_transfers
    assert transfer not in scheduler2.incoming_transfers
    # Verify that the running totals have been updated.
    assert scheduler1.allocated_outgoing == 0
    assert scheduler2.allocated_incoming == 0
    # Verify that the finish event has been removed.
    assert not any(event.data.get("transfer") == transfer for (_, _, event) in client1.simulator.events)


def test_kill_transfer_pending(setup_schedulers):
    scheduler1, scheduler2, _, _ = setup_schedulers
    # Create a transfer that is pending (i.e. not yet active)
    transfer = Transfer(scheduler1, scheduler2, 1000, "test_model", {})
    scheduler1.outgoing_requests.append(transfer)
    scheduler2.incoming_requests.append(transfer)
    
    scheduler1.kill_transfer(transfer)
    
    # Verify that the transfer is removed from the pending lists.
    assert transfer not in scheduler1.outgoing_requests
    assert transfer not in scheduler2.incoming_requests


def test_kill_all_transfers(setup_schedulers):
    scheduler1, scheduler2, _, _ = setup_schedulers

    # Create an active transfer where scheduler1 is the sender.
    transfer1 = scheduler1.add_transfer(scheduler2, 1000, "test_model", {})
    transfer2 = scheduler2.add_transfer(scheduler1, 1000, "test_model", {})

    # Create pending transfers for scheduler1.
    pending_transfer1 = Transfer(scheduler1, scheduler2, 1000, "test_model", {})
    scheduler1.outgoing_requests.append(pending_transfer1)
    scheduler2.incoming_requests.append(pending_transfer1)
    
    pending_transfer2 = Transfer(scheduler2, scheduler1, 1000, "test_model", {})
    scheduler2.outgoing_requests.append(pending_transfer2)
    scheduler1.incoming_requests.append(pending_transfer2)
    
    # Kill all transfers on scheduler1.
    scheduler1.kill_all_transfers()
    
    # Verify that scheduler1's active and pending lists are empty.
    for scheduler in [scheduler1, scheduler2]:
        assert not scheduler.outgoing_transfers
        assert not scheduler.incoming_transfers
        assert not scheduler.outgoing_requests
        assert not scheduler.incoming_requests


logging.basicConfig(level=logging.DEBUG)
