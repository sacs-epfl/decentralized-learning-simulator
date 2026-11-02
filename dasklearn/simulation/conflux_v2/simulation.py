import csv
import os
import math
from dasklearn.simulation.simulation import Simulation
from dasklearn.simulation.conflux_v2.client import ConfluxV2Client
from dasklearn.simulation.events import START_ROUND, FINISH_TRAIN


class ConfluxV2Simulation(Simulation):
    CLIENT_CLASS = ConfluxV2Client

    def __init__(self, settings):
        if settings.churn != "none":
            raise ValueError("Conflux v2 currently requires stable membership (use --churn none).")
        if not 0 < settings.success_fraction <= 1 or settings.sample_size < 1 or settings.chunks_in_sample < 1 or settings.relay_count < 0:
            raise ValueError("Invalid Conflux v2 sample, chunk, relay or success-fraction setting")
        if settings.test_method == "global":
            raise ValueError("Use --test-method snapshot or individual for Conflux v2")
        if settings.server_optimizer not in {"fedavg", "fedavgm", "fedadam"}:
            raise ValueError("Unknown Conflux server optimizer")
        if not all(math.isfinite(v) for v in [settings.server_learning_rate,settings.server_beta1,
                settings.server_beta2,settings.server_tau,settings.server_step_ms,settings.state_pack_ms]):
            raise ValueError("Server optimizer settings must be finite")
        if settings.server_learning_rate<=0 or not 0<=settings.server_beta1<1 or not 0<=settings.server_beta2<1 or settings.server_tau<=0:
            raise ValueError("Invalid server learning rate, decay, or tau")
        if settings.reducer_epoch_rounds<0 or settings.state_bits not in {8,32} or settings.state_block_size<1 or min(settings.server_step_ms,settings.state_pack_ms)<0:
            raise ValueError("Invalid optimizer state placement, precision, or compute cost")
        if settings.server_optimizer=='fedavg' and settings.server_learning_rate!=1:
            raise ValueError("FedAvg uses server_learning_rate=1; select fedavgm with beta1=0 for server SGD")
        if settings.server_optimizer!='fedavg' and settings.lora:
            raise ValueError("Reducer FedOpt does not yet support LoRA state layouts")
        settings.sample_size = min(settings.sample_size, settings.participants)
        super().__init__(settings)
        if settings.server_optimizer!='fedavg':
            from dasklearn.simulation.conflux_v2.fedopt_client import ConfluxFedOptClient
            self.CLIENT_CLASS=ConfluxFedOptClient
        self.contributor_sets = {}
        self.completed_rounds = []
        self.optimizer_handoffs = []
        self.register_event_callback(START_ROUND, "start_round")
        self.register_event_callback(FINISH_TRAIN, "finish_train")

    def initialize_clients(self):
        if self.settings.server_optimizer!='fedavg':
            from dasklearn.functions import initialize
            model=initialize(self.settings,{})[0]
            self.optimizer_elements=sum(value.numel() for value in model.state_dict().values())
            if self.settings.chunks_in_sample>self.optimizer_elements:
                raise ValueError("Optimizer chunks must contain at least one model-state element")
        capacities = {client.index: client.bw_scheduler.bw_limit for client in self.clients}
        if min(capacities.values()) <= 0:
            raise ValueError("Conflux v2 requires positive bandwidth capacities")
        for client in self.clients:
            client.other_nodes_bws = capacities.copy()
        super().initialize_clients()

    def save_measurements(self):
        super().save_measurements()
        with open(os.path.join(self.data_dir, "conflux_v2_rounds.csv"), "w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["round", "client", "time_us", "contributors"])
            for round_nr, client, time in self.completed_rounds:
                writer.writerow([round_nr, client, time, " ".join(map(str, self.contributor_sets[round_nr]))])
        if self.settings.server_optimizer!='fedavg':
            with open(os.path.join(self.data_dir,"conflux_optimizer_handoffs.csv"),"w") as stream:
                writer=csv.writer(stream)
                writer.writerow(["source_round","chunk","from","to","payload_bytes"])
                writer.writerows(self.optimizer_handoffs)
