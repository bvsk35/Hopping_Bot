#!/usr/bin/env python3

import pickle
import logging

import numpy as np


LOGGER = logging.getLogger(__name__)


class SampleList(object):
	""" Class that handles writes and reads to sample data. 
	Args: Sample object storing the data This will c
	"""
	def __init__(self, samples):
		self._samples = samples

	def get_X(self, idx=None):
		""" Returns N x T x dX numpy array of states. 
			This function returns all the X values for all the condition
		"""
		if idx is None:
			idx = range(len(self._samples))
		return np.asarray([self._samples[i].get_X() for i in idx])

	def get_U(self, idx=None):
		""" Returns N x T x dU numpy array of actions. 
			This function returns all the U values for all the condition
		"""
		if idx is None:
			idx = range(len(self._samples))
		return np.asarray([self._samples[i].get_U() for i in idx])

	def get_noise(self, idx=None):
		""" Returns N x T x dU numpy array of noise generated during rollouts. 
			This function returns all the noise applied in the all the condition
		"""
		if idx is None:
			idx = range(len(self._samples))
		return np.asarray([self._samples[i].get('NOISE') for i in idx])

	def get_obs(self, idx=None):
		""" Returns N x T x dO numpy array of features. 
			This function returns all the Obs values from the all the condition
		"""
		if idx is None:
			idx = range(len(self._samples))
		return np.asarray([self._samples[i].get_obs() for i in idx])

	def get_samples(self, idx=None):
		""" Returns N sample objects. 
			This will return all the sample object for all the condition
		"""
		if idx is None:
			idx = range(len(self._samples))
		return [self._samples[i] for i in idx]

	def num_samples(self):
		""" Returns number of samples. 
			This could be interepreted as number of conditions
		"""
		return len(self._samples)

	# Convenience methods.
	def __len__(self):
		return self.num_samples()

	def __getitem__(self, idx):
		return self.get_samples([idx])[0]


class PickleSampleWriter(object):
	""" Pickles samples into data_file. """
	def __init__(self, data_file):
		self._data_file = data_file

	def write(self, samples):
		""" Write samples to data file. """
		with open(self._data_file, 'wb') as data_file:
			cPickle.dump(data_file, samples)


class SysOutWriter(object):
	""" Writes notifications to sysout on sample writes. """
	def __init__(self):
		pass

	def write(self, samples):
		""" Write number of samples to sysout. """
		LOGGER.debug('Collected %d samples', len(samples))
