from mujoco_py import load_model_from_path, MjSim, MjViewer
import mujoco_py
import numpy as np
'''
Mujoco is a simulator of choice,
Please visit mujoco: http://www.mujoco.org/ for the mujoco installation.
Also do pip3 install mujoco-py
'''

class Mujoco_sim():
    def __init__(self, path, viewer = False):
        Model = path
        assert isinstance(Model, str)
        model_loaded = load_model_from_path(Model)
        self.sim = MjSim(model_loaded)
        print("model added to the simulator")
        self.CondViewer = viewer

    def load(self,state_vector,control_vector,K,k,input,initial):
        '''
        This function loads 
        1. the inital position of the simulator.
        3. get the state and cotnrol vectors (xs,us).
        2. gets the controller values (K,k).
        dimension:
        K = (N,xD)
        k  = (N,uD)
        us = (N,uD)
        xs = (N,xD)
        '''
        N,xD = state_vector.shape
        N,uD = control_vector.shape

        if initial:
            self.sim.data.qpos[:] = 0
            self.sim.data.qvel[:] = 0
        else:
            #manually add the initial condition,
            self.sim.data.qpos[0] = input[0]
            self.sim.data.qpos[1] = input[1]
            self.sim.data.qvel[0] = input[2]
            self.sim.data.qvel[1] = input[3]
        
        assert K.shape == (N,uD,xD)
        assert k.shape == (N,uD)
        self.K = K
        self.k = k
        self.xs = state_vector
        self.us = control_vector
        self.N = N
        self.total_controls = []
        self.state = []


    def runSimulation(self, getData = False):
        '''
        This function is to run the simulation
        GetData: boolean ensure you get the data of the simulator after the simulation
        '''
        if self.CondViewer:
            viewer = mujoco_py.MjViewer(self.sim)
        for i in range(self.N):
            if i == 0:
                self.state = np.append(self.sim.data.qpos,self.sim.data.qvel)
                self.state = np.expand_dims(self.state,1)
            else:
                
                self.state = np.append(state, 
                            np.expand_dims(np.append(self.sim.data.qpos,self.sim.data.qvel),1),axis = 1)
            #the following lines are specific for the cartpole
            #TODO: generalize this to all the simulation
            current = np.c_[self.sim.data.qpos[0],self.sim.data.qvel[0],np.sin(self.sim.data.qpos[1]),
                  np.cos(self.sim.data.qpos[1]),self.sim.data.qvel[1]].T
            # TODO: Send param switch to change control signal from us to feedback control signal
            control = np.dot(self.K[i,:],(self.xs[i].reshape(5,1) - current))  + self.k[i] + self.us[i]
            self.sim.data.ctrl[0] = self.us[i]
            self.sim.forward()
            self.sim.step()
            self.total_controls.append(control)

            if self.CondViewer:
                viewer.render()
            
        if getData:
            print('return the state and dimensions')
            return self.state

    def get_U(self, t=None):
        """ This will give out all (or t) the control actions applied to the simulator """
        if t == None:
            return self.total_controls
        else:
            return self.total_controls[:t]
    
    def get_X(self, t=None):
        """ This will give out all (or t) the states including velocity of the system """
        if t == None:
            return self.state
        else:
            return self.state[:,:t]
        
    def getObs(self, t=None):
        """ This will give out all (or t) the observable states in the system """
        if t == None:
            return self.state[:2,:]
        else:
            return self.state[:2,:t]