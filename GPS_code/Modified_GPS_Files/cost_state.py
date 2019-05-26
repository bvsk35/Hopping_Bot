import copy
import numpy as np
from hyperparameters import Agent_params
from hyperparameters import Cost_state
from cost_utils import cost_utils

class CostState():
    '''
    This class is to calculate the state cost of the sample.
    '''
    def __init__(self, hyperparams): #need to send cost state as hyperparams
        config = copy.deepcopy(hyperparams)
        config.update(hyperparams)
        self._hyperparams = config #basically the cost_state dictionary
        self.Agent_params = Agent_params
        self.utils = cost_utils

    def eval(self,sample):
        '''
        Evaluate cost function and derivatives on a sample
        Args:
            object of the sample
        '''
        T = sample.T
        Du = sample.dU
        Dx = sample.dX
        # creating empty cost ndarrays
        final_l = np.zeros(T)
        final_lu = np.zeros((T, Du))
        final_lx = np.zeros((T, Dx))
        final_luu = np.zeros((T, Du, Du))
        final_lxx = np.zeros((T, Dx, Dx))
        final_lux = np.zeros((T, Du, Dx))

        for data_type in Agent_params:
            wp = self.Agent_params[data_type]['wp']
            tgt = self.Agent_params[data_type]['target_state']
            x = sample.get(data_type)
            _,dim_sensor = x.shape

            wpm = self.utils.get_ramp_multiplier(self.utils.RAMP_CONSTANT, T)

            wp = wp * np.expand_dims(wpm, axis=-1)
            # Compute state penalty.
            dist = x - tgt

            # Evaluate penalty term.
            l, ls, lss = self.utils.evall1l2term(
                wp, dist, np.tile(np.eye(dim_sensor), [T, 1, 1]),
                np.zeros((T, dim_sensor, dim_sensor, dim_sensor)),
                self._hyperparams['l1'], self._hyperparams['l2'],
                self._hyperparams['alpha']
            )
            # computing the final cost
            final_l += l
            # adding data to the sample through function in the agent
            sample.agent.pack_data_x(final_lx,ls,data_types=[data_type])
            sample.agent.pack_data_x(final_lxx,lss,data_types=[data_type, data_type])

        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux