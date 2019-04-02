# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:19:19 2019

@author: defaultuser0
"""

#SMOTE算法及其python实现
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
class Smote:
    def __init__(self,samples,N=10,k=5):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0
        self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))

    def over_sampling(self):
#        print(11111111111)
#        N=int(self.N/100)
        N=int(self.N)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
#        print ('neighbors',neighbors)
        for i in range(len(self.samples)):
#            print('samples',self.samples[i])
            nnarray=neighbors.kneighbors(self.samples[i].reshape((1,-1)),return_distance=False)[0]  #Finds the K-neighbors of a point.
#            print ('nna',nnarray)
            self._populate(N,i,nnarray)
        return self.synthetic


    # for each minority class sample i ,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self,N,i,nnarray):
#        print(2222222222)
        for j in range(N):
#            print('j',j)
            nn=random.randint(0,self.k-1)  #包括end
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1
#            print(self.newindex)
# a=np.array([[1,2,3],[4,5,6],[2,3,1],[2,1,2],[2,3,4],[2,3,4]])
# s=Smote(a,N=10)
# b=s.over_sampling()
