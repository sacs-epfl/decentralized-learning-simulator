from dasklearn.simulation.motef.client import MoTEFClient


class MoTEFVRClient(MoTEFClient):
    BOOTSTRAP_OPERATION = 'motef_vr_bootstrap'
    GRADIENT_OPERATION = 'motef_vr_gradient'
    GRADIENT_EVALUATIONS = 2

    def gradient_inputs(self, state):
        return {**super().gradient_inputs(state), 'previous_model': state['base']}
