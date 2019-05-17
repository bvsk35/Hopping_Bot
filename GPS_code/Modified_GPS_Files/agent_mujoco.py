#!/usr/bin/env python3

#import basic packages
import numpy as np
import copy
import mujoco_py as mj
from sample import Sample

#import file dependent packages
from hyperparameters import Agent_params
from agent_new import Agent

class AgentMujoco(Agent):
	def __init__(self,hyperparameters):
		config_agent = copy.deepcopy(hyperparameters)
		Agent.__init__(self,config_agent)
		self._hyperparameters = config_agent
		# self._setup_conditions() TODO
		self._setup_sim(config_agent['world_path'])
		self._set_initial_pos()

	def _setup_conditions(self):
		''' 
		TODO: add robost things for varying conditions.
		'''

	def _setup_sim(self,path_to_world,verbose=False):
		'''
		This function is to setup the mujoco world and initialiase the states of the work for
		the specific conditions
		Args:
			file path to the world
		'''
		self._sim = []
		
		if len(path_to_world) == 1:
			self._sim = mj.MjSim(mj.load_model_from_path(path_to_world[1]))
			if self._hyperparameters['render']:
				self._viewer = mj.MjViewer(self._sim)
		else:
			self._sim = [mj.MjSim(mj.load_model_from_path(path)) for i,path in enumerate(path_to_world)]
			if self._hyperparameters['render']:
				self._viewer = [mj.MjViewer(self._sim[i]) for i,path in enumerate(path_to_world)]

		if verbose:
			print('#'*30)
			print('loading the model from the path' + path_to_world)

	def _set_initial_pos(self):
		try:
			assert self._hyperparameters['conditions'] != 1
		except:
			raise AssertionError('condition is one please change it!!!!')
		for i in range(self._hyperparameters['conditions']):
			self.x0 = self._hyperparameters['initial_condition']

	def sample(self, policy, condition, verbose=False, save=True, noisy=False):
		self._set_initial_pos()
		self._init(condition)
		new_sample = self._init_sample(condition)
		U = np.zeros([self.T,self.dU])
		if noisy:
			#TODO: define noise somewhere
			pass
		else:
			noise = np.zeros((self.T,self.du))
		# This is the main control action loop
		for t in range(self.T):
			X_t = new_sample.get_X(t=T)
			obs_t = new_sample.get_obs(t=T)
			U[t,:] = policy.act(X_t, Obs_t, t, noise[t,:])
			if (t+1) < self.T:
				# if 'substeps' in self._hyperparameters['substeps']:
				#IMPORTANT: mucj have substeps = 1
				for _ in range(self._hyperparameters['substeps']):
					self._sim[condition].ctrl[:] = U[t, :]
					self._sim[condition].step()
				Mj_X = self._sim[condition].get_states()
				self._set_sample(new_sample,Mj_X,t,condition,feature_fn=False)
			new_sample.set('ACTION',U)
			if save:
				self._samples[condition].append(new_sample)
			return new_sample

	def reset(self, condition):
		self._samples[condition] = []

	def _init(self, condition):
		'''
		Set the world to the initial position and run the step
		Args:
			Condition
		'''
		for i in self._hyperparameters['condition']:
			self._sim[i].set_state(mj.MjSimState(time=0.0, qpos=self.x0[int(len(self.x0)/2):],
				qvel=self.x0[:int(len(self.x0)/2)]), act=None, udd_state={})

	def _init_sample(self, condition):

		sample = Sample(self)

		self._init(condition)

		data = self._sim[condition].get_states()

		sample.set('TORSO_POSTION',data['qpos'][:2],t=0)
		sample.set('TORSO_VELOCITIES',data['qvel'][:2],t=0)
		sample.set('JOINT_ANGLES',data['qpos'][2:],t=0)
		sample.set('JOINT_VELOCITIES',data['qvel'][2:],t=0)

		return sample
		#we can also get the jacobian for the points using the 
		# sim.data.get_site_jabp and sim.data.get_site_jabr
		#in the example they have set the end effector points
		#they have stored the images data as well
		# a = sim.render(width=200, height=200, camera_name='fixed', depth=True)
		## a is a tuple if depth is True and a numpy array if depth is False ##

	def _set_sample(self,sample,mj_X,t,condition,feature_fn=None):
		'''
			This is used to append the data after each iteration
			Args:
				sample: Sample object to set data for.
				mj_X: Data to set for sample.
				t: Time step to set for sample.
				condition: Which condition to set.
				feature_fn: function to compute image features from the observation.  
		'''
		
		sample.set('TORSO_POSTION',mj_X['qpos'][:2],t=t+1)
		sample.set('TORSO_VELOCITIES',mj_X['qvel'][:2],t=t+1)
		sample.set('JOINT_ANGLES',mj_X['qpos'][2:],t=t+1)
		sample.set('JOINT_VELOCITIES',mj_X['qvel'][2:],t=t+1)
		#TODO: add sensor to the hopper3 and get the jacobian workings
		#Where ever there is site in the xml file you can find jacobian

	def _get_image_from_obs(self,obs):
		pass
