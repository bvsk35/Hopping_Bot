import numpy as np
import abc

from sample import Sample
from sample_list import SampleList
class Agent(object):
    '''
    Agent superclass. The agent interacts with the environment to collect samples
    '''
    __metaclass__ = abc.ABCMeta
    def __init__(self,hyperparams):
        self._hyperparams = hyperparams
        Agent = self._hyperparams['Agent_params']
        self._samples = [[] for i in range(Agent['No_condition'])]
        self.T = Agent['Time']
        self.dU = self._hyperparams['sensor_dims']['ACTION']

        self.x_data_types = self._hyperparams['state_include']
        self.obs_data_types = self._hyperparams['obs_include']
        if 'meta_include' in self._hyperparams:
            self.meta_data_types = self._hyperparams['meta_include']
        else:
            self.meta_data_types = []
        # List of indices for each data type in state X.
        self._state_idx, i = [], 0
        for sensor in self.x_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._state_idx.append(list(range(i, i+dim)))
            i += dim
        self.dX = i
        
        self._obs_idx, i = [], 0
        for sensor in self.obs_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._obs_idx.append(list(range(i, i+dim)))
            i += dim
        self.dO = i

        # List of indices for each data type in meta data.
        self._meta_idx, i = [], 0
        for sensor in self.meta_data_types:
            dim = self._hyperparams['sensor_dims'][sensor]
            self._meta_idx.append(list(range(i, i+dim)))
            i += dim
        self.dM = i

        self._x_data_idx = {d: i for d, i in zip(self.x_data_types,
                                                 self._state_idx)}
        self._obs_data_idx = {d: i for d, i in zip(self.obs_data_types,
                                                   self._obs_idx)}
        self._meta_data_idx = {d: i for d, i in zip(self.meta_data_types,
                                                   self._meta_idx)}

    @abc.abstractmethod
    def sample(self, policy, condition, verbose=False, save=True, noisy=False):
        '''
        Get the samples from the model for the give policy
        '''
        raise NotImplementedError("Must be implemented in subclass.")

    @abc.abstractmethod
    def reset(self, condition):
        ''' 
        Reset the system into the given initial condition
        '''
        raise NotImplementedError("Must be implemented in subclass.")


    def get_sample(self, condition,start=0,end=None, SampleObject=False):
        '''
            Output the data from the start to end of the system for a give condition
        '''
        if SampleObject:
            return (SampleList(self._sample[condition][start:]) if end == None 
                else SampleList(self._sample[condition][start:end]))
        else:
            return (self._sample[condition][start:] if end == None 
                else self._sample[condition][start:end])

    def clear_samples(self,condition=None):
        '''
        This function is used to clear the specific or all samples of the agent
        '''
        if condition == None:
            self._sample = [[] for i in range(Agent['No_rollouts'])] 
        else:
            self._samples[condition] = []

    def delete_last_sample(self,condition):
        '''
        Delete the last sample.
        '''
        self._sample[condition].pop()

    def get_idx_x(self, sensor_name):
        """
        Return the indices corresponding to a certain state sensor name.
        Args:
            sensor_name: The name of the sensor.
        """
        return self._x_data_idx[sensor_name]

    def pack_data_x(self, existing_mat, data_to_insert, data_types, axes=None):
        """
        Update the state matrix with new data.
        Args:
            existing_mat: Current state matrix.
            data_to_insert: New data to insert into the existing matrix.
            data_types: Name of the sensors to insert data for.
            axes: Which axes to insert data. Defaults to the last axes.
        """
        num_sensor = len(data_types)
        if axes is None:
            # If axes not specified, assume indexing on last dimensions.
            axes = list(range(-1, -num_sensor - 1, -1))
        else:
            # Make sure number of sensors and axes are consistent.
            if num_sensor != len(axes):
                raise ValueError(
                    'Length of sensors (%d) must equal length of axes (%d)',
                    num_sensor, len(axes)
                )