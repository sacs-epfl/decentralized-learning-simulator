"""Algorithm 2 shares Algorithm 1's causal compressed exchange protocol."""
import torch

from dasklearn.simulation.motef.simulation import MoTEFSimulation
from dasklearn.simulation.motef_vr.client import MoTEFVRClient


class MoTEFVRSimulation(MoTEFSimulation):
    CLIENT_CLASS = MoTEFVRClient
    MEASUREMENT_PREFIX = 'motef-vr'

    def __init__(self, settings):
        if torch.device(settings.torch_device_name).type not in {'cpu', 'cuda'}:
            raise ValueError('MoTEF-VR requires CPU or CUDA for coupled stochastic RNGs')
        super().__init__(settings)
