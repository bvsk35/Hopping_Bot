#!/usr/bin/env python3
import numpy as np
from sklearn import mixture

class local_estimate:
    #well here instead of the local policy I am taking in data
    #data dimension is 10X5
    def __init__(self,data,components,NoOfstates,lam = 0):
        self.lam = lam
        self.numberOfStates = NoOfstates
        self.N = NoOfstates * 2 + 1
        self.data = data
        self.components = components
        self.gmm_data = self.data
        self.means = None
        self.covariances = None
        self.NIW_id = 0
                                                

    def estimate(self,gmm_true=True,N=500):
        self.gmm_true = gmm_true
        if self.gmm_true:
            x,y = self.GMM_method()
            x,y = self.NIW_distribution(x,y,N)
            A,B,C = self.Post_predict(x,y)
            return(A,B,C)
        else:
            return self.old()

    def append_data(self,data):
        if self.means == None:
            self.data = data
        else:
            self.data = np.append(self.data,data,axis = 0)
        self.gmm_data = self.data

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


    def old(self):
        self.means = np.mean(self.data,0)
        for x in self.data.T:
            x = x.T
            self.covariances += np.dot((x-self.means),(x-self.means).T)
        self.covariances = self.covariances / (len(self.data) -1)

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