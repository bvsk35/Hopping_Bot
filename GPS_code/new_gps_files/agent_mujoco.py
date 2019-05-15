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
        pass
    
    def reset(self, condition):
        pass

    def _init(self, condition):
        '''
        Set the world to the initial position and run the step
        Args:
            Condition
        '''
        for i in self._hyperparameters['condition']:
            self._sim[i].set_state(mj.MjSimState(time=0.0, qpos=self.x0[int(len(self.x0)/2):],
                qvel=self.x0[:int(len(self.x0)/2)]), act=None, udd_state={}))

    def _init_sample(self, condition):
        sample = Sample(self)

        self._init(condition)

        data = self._sim[condition].get_states()

        sample.set()