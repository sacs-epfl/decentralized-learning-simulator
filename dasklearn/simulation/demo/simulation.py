"""Stable-membership DeMo with an explicit decentralized ring all-gather."""
import math
from dasklearn.models import create_model
from dasklearn.optimizer.demo import packet_bytes
from dasklearn.simulation.simulation import Simulation
from dasklearn.simulation.events import FINISH_TRAIN
from dasklearn.simulation.demo.client import DeMoClient


class DeMoSimulation(Simulation):
    CLIENT_CLASS = DeMoClient

    def __init__(self, settings):
        if settings.churn != 'none':
            raise ValueError('DeMo collective currently requires stable membership')
        if settings.learning.momentum:
            raise ValueError('DeMo uses its private compression residual; set SGD momentum=0')
        if not math.isfinite(settings.compression_decay) or not 0 <= settings.compression_decay < 1:
            raise ValueError('DeMo decay must be finite in [0,1)')
        if settings.compression_chunk < 1 or settings.compression_topk < 1 or settings.learning.local_steps < 1:
            raise ValueError('Positive chunk, topk and gradient accumulation count required')
        if settings.test_method == 'global':
            raise ValueError('Use snapshot or individual evaluation')
        model = create_model(settings.dataset, architecture=settings.model, settings=settings)
        if list(model.buffers()) or any(not p.requires_grad for p in model.parameters()):
            raise ValueError('DeMo currently requires models without buffers or frozen parameters')
        self.packet_size = packet_bytes([p.shape for p in model.parameters()], settings.compression_chunk,
                                       settings.compression_topk, next(model.parameters()).element_size())
        self.updates = {}
        super().__init__(settings)
        self.register_event_callback(FINISH_TRAIN, 'finish_train')
