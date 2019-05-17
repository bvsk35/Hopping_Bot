import copy
import numpy as np


class CostState():
    '''
    This class is to calculate the state cost of the sample.
    '''
    def __init__(self, hyperparams):
        config = copy.deepcopy(hyperparams)
        config.update(hyperparams)
        self._hyperparams = config

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

        for data_type in self._hyperparams[]