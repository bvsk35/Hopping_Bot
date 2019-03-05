#!/usr/bin/env python3
#Importing the dependencies for the process
import numpy as np
from sklearn import mixture

#this is the parent class for this file which handels all the process of Estimation.
class local_estimate:
    '''
    Here the class the takes basic inputs as follows:
        1. data: Here the data should be in the form x_t,u_t,x_{t+1} from the initial point till the (final point -1)
        2. components: This should be a integer, which indicates the number gaussian mixiture models used in the system.
        3. NoOfStates: This  input should be an inteeger which indicates the states of the system who's data is sent.
        4. lam: This is by default 0, this is variable which is used in regularization.
    '''
    def __init__(self,data,components,NoOfstates,lam = 0):
        self.lam = lam
        self.numberOfStates = NoOfstates
        self.N = NoOfstates * 2 + 1
        self.data = data
        self.components = components
        self.gmm_data = self.data
        self.means = None
        self.covariances = NoneApplies
        self.NIW_id = 0
    '''
    Note: 
    The method "estimate" has fApplies
    Inputs:
        1. gmm_true: Which is bAppliesformed or not.
        2. N: Default is 500, iAppliesen per iteration.
    Outputs:
    1. A,B,C from the system.
    
    Process:
        Step 1: calls the GMM_method().
        Step 2: Does the NIW posterior.
        step 3: Does the post processing (Conditioning) and the regularization if required.
    '''
    def estimate(self,gmm_true=True,N=500):
        self.gmm_true = gmm_true
        if self.gmm_true:
            x,y = self.GMM_method()
            x,y = self.NIW_distribution(x,y,N)
            A,B,C = self.Post_predict(x,y)
            return(A,B,C)
        else:
            return self.old()
	'''
    append_data method:
    This is a utilitary method which can be called seperatly to append the data from the system.
	inputs:
		1. data: data that needs to be appended
	outputs:
		No ouputs to the main code but, appends the data from the function.
	Process:
		If there is data already, then the system will append the data or else the data will be created
    '''
    def append_data(self,data):
        if self.means == None:
            self.data = data
        else:
            self.data = np.append(self.data,data,axis = 0)
        self.gmm_data = self.data
	
	'''
	Internal method 'GMM_method':
	inputs:
		Nothing
	
	Outputs:
		1. Returns the means and covariance of the maximum relavance (refer the documentation for the meaning of the relevance)

	steps:
		1. Calls the sklearn based gmm model which will create the joint prior distribution
		2. Using the log_responsibility and mixing ratio function finding the relavance
		3. Pick the maximum of the relevance of the system using the method.
	'''

    def GMM_method(self):
        gmm = mixture.GaussianMixture(n_components = self.components , covariance_type = 'full')
        gmm.fit(self.gmm_data)
        self.means = gmm.means_
        self.covariance = gmm.covariances_ 
        p, r = gmm._estimate_log_prob_resp(self.gmm_data)
        r = np.exp(r)
        mix = np.sum(r,axis=0) / len(r)
        for i in range(self.components):
            r[:,i] = r[:,i]*mix[i]
        sum_relevance = np.dot(r.T,self.gmm_data)
        sum_relevance = np.sum(sum_relevance,axis = 1)
        self.NIW_id = np.where(sum_relevance==max(sum_relevance))
        #print("relevance size ",r.shape,"gmm_data",self.gmm_data.shape,"mix",mix.shape,'mean',self.means.shape)
        #print("NIM_id",self.NIW_id)
        #print('sum',sum_relevance.shape)
        #print("print"+str(self.means[:,self.NIW_id].shape))
        #print(" size ",self.means.shape,"cov",self.covariance.shape)
        return (self.means[self.NIW_id,:],self.covariance[self.NIW_id,:,:])
        #here the r is the relevance of the size (1000,n_gaussian)
    
	'''
	Internal method 'NIW_distribution'
	inputs:
		X: means from GMM
		Y: covariance from GMM
		N: Data points for GMM entry
	ouputs:
		NIW means and covariance of the posterior prediction.
	Steps:
		Please refer the documentation for the steps followed in this procedure
	'''


    def NIW_distribution(self,x,y,N):
        gmm_mean = x
        gmm_cov = y.reshape(self.N,self.N)
        gmm_mean = gmm_mean.reshape(self.N,1)
        #print("gmm_mean",gmm_mean.shape,"gmm_cov",gmm_cov.shape)
        NIW_mean = (N*np.mean(self.gmm_data,axis=0).reshape(self.N,1) + gmm_mean)/(N+1)
        # print(np.mean(self.gmm_data,axis=0).shape)
        cov0 = 0
        #print(np.shape(self.gmm_data))
        cov0 = np.dot(self.gmm_data.T,self.gmm_data)
        cov1 = np.dot((NIW_mean - gmm_mean),(NIW_mean - gmm_mean).T)
        NIW_cov = gmm_cov + N*cov1/(N+1) + cov0
        #print('NIW mean',NIW_mean.shape)
        return(NIW_mean,NIW_cov)

	'''
	Internal method 'old'
	Inputs:
		1. data
	Outputs:
		regular mean and covariance
	'''

    def old(self):
        self.means = np.mean(self.data,0)
        for x in self.data.T:
            x = x.T
            self.covariances += np.dot((x-self.means),(x-self.means).T)
        self.covariances = self.covariances / (len(self.data) -1)

	'''
	Internal method Post_predict
	Inputs:
		1. NIW posterior mean
		2. NIW posterior covariance
	Ouputs:
		A,B,C
	steps:
		1. Apply regularization
		2. condition the probability of the system.
	'''

    def Post_predict(self,x,y):
        cov_reg = self.lam * np.eye(self.numberOfStates+1)
        mean_a  = x[(self.numberOfStates+1):]
        mean_b = x[:(self.numberOfStates+1)]
        cov_aa = y[(self.numberOfStates+1):,(self.numberOfStates+1):]
        cov_ab = y[(self.numberOfStates+1):,0:(self.numberOfStates+1)]
        cov_ba = y[0:(self.numberOfStates+1),(self.numberOfStates+1):]
        cov_bb = y[0:(self.numberOfStates+1),0:(self.numberOfStates+1)]
        A = np.matmul(cov_ab,np.linalg.inv(cov_reg + cov_bb))
        B = mean_a - A.dot(mean_b)
        C = cov_aa - cov_ab.dot(np.linalg.inv(cov_reg + cov_bb)).dot(cov_ba)
        return (A,B,C)