import numpy as np
#IMPORTANT: the helper fnc pack_data_x and pack_data_obs are in agent file for some
#weird reason In future we have to see how to add in the sample class and not in the 
#agent class
class Sample(object):
    """
    Class that handles the representation of a trajectory and stores a
    single trajectory.    
    Args:
        Agent Object
    Usage: created and used in Agent_new.py
    """
    def __init__(self, agent):
        #gathering all the infomation about the system.
        self.agent = agent
        self.T = agent.T
        self.dX = agent.dX
        self.dU = agent.dU
        self.dO = agent.dO
        self.dM = agent.dM
        # Dictionary containing the sample data from various sensors.
        self._data = {} #This will be the main data storing dictionary
        # Creating the _X, _Obs and _meta files 
        self._X = np.empty((self.T, self.dX))
        self._X.fill(np.nan)
        self._obs = np.empty((self.T, self.dO))
        self._obs.fill(np.nan)
        self._meta = np.empty(self.dM)
        self._meta.fill(np.nan)

    def set(self, sensor_name, sensor_data, t=None):
        """ Set trajectory data for a particular sensor. """
        # If the time is None then the data according to the sensor name is dumped in _data
        #  Else is stores in the _data after creating the proper dimensions
        if t is None:
            self._data[sensor_name] = sensor_data
            self._X.fill(np.nan)  # Invalidate existing X.
            self._obs.fill(np.nan)  # Invalidate existing obs.
            self._meta.fill(np.nan)  # Invalidate existing meta data.
        else:
            if sensor_name not in self._data:
                self._data[sensor_name] = \
                        np.empty((self.T,) + sensor_data.shape)
                self._data[sensor_name].fill(np.nan)
            self._data[sensor_name][t, :] = sensor_data
            self._X[t, :].fill(np.nan)
            self._obs[t, :].fill(np.nan)

    def get(self, sensor_name, t=None):
        """ Get trajectory data for a particular sensor. 
            Outputs the given data for particular sensor
        """
        return (self._data[sensor_name] if t is None
                else self._data[sensor_name][t, :])

    def get_X(self, t=None):
        """ Get the state. Put it together if not precomputed. 
            Send the data in the self._X if the data is NAN
            Get the all the state data from the self._data file
            and then send it to the agent to pack the data and then return it
        """
        X = self._X if t is None else self._X[t, :]
        if np.any(np.isnan(X)):
            for data_type in self._data:
                if data_type not in self.agent.x_data_types:
                    continue
                data = (self._data[data_type] if t is None
                        else self._data[data_type][t, :])
                self.agent.pack_data_x(X, data, data_types=[data_type])
        return X

    def get_U(self, t=None):
        """ Get the action. """
        return self._data['ACTION'] if t is None else self._data['ACTION'][t, :]

    def get_obs(self, t=None):
        """ Get the Obs. Put it together if not precomputed. 
            Send the data in the self._Obs if the data is NAN
            Get the all the state data from the self._data file
            and then send it to the agent to pack the data and then return it
        """
        obs = self._obs if t is None else self._obs[t, :]
        if np.any(np.isnan(obs)):
            for data_type in self._data:
                if data_type not in self.agent.obs_data_types:
                    continue
                if data_type in self.agent.meta_data_types:
                    continue
                data = (self._data[data_type] if t is None
                        else self._data[data_type][t, :])
                self.agent.pack_data_obs(obs, data, data_types=[data_type])
        return obs

    def get_meta(self):
        #TODO: figure out what meta means to the controller 
        # this function will not be used
        """ Get the meta data. Put it together if not precomputed. """
        meta = self._meta
        if np.any(np.isnan(meta)):
            for data_type in self._data:
                if data_type not in self.agent.meta_data_types:
                    continue
                data = self._data[data_type]
                self.agent.pack_data_meta(meta, data, data_types=[data_type])
        return meta

    # For pickling.
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('agent')
        return state

    # For unpickling.
    def __setstate__(self, state):
        self.__dict__ = state
        self.__dict__['agent'] = None