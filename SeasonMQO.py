import numpy as np 
from numpy.linalg import inv

class MQOSeasonal :

    def __init__(self,period):
        self.s = period
        
    def filter_dummie(self,y): 
        X = self.__dumie_matrix(y)
        return self.__filter(X,y)

    def filter_trig(self,y): 
        X = self.__trig_matrix(y)
        return self.__filter(X,y)

    def __filter(self,X,y):
        Y = np.matrix(y).T
        P = inv(X.T*X)*X.T*Y
        Y_filtered = X*P 
        return Y_filtered.T

    def __dumie(self,j,t):
        if t % self.s == 0:
            return -1
        if t % self.s == j:
            return 1
        return 0

    def __harmonic(self,j,t):
        season = j % self.s
        harmonic_order = (season // 2) + 1
        harmonic_type = season % 2
        if harmonic_type == 0:
            return np.cos(2*np.pi*harmonic_order*t/self.s)
        if harmonic_type == 1:
            return np.sin(2*np.pi*harmonic_order*t/self.s)

    def __dumie_matrix(self,y):
        dummie_form = [ \
            [1]+[self.__dumie(j,t) for j in range(1,self.s)] \
            for t in range(len(y))]
        return np.matrix(dummie_form)
    
    def __trig_matrix(self,y):
        trig_form = [ \
            [1] + [self.__harmonic(j,t) for j in range(self.s-1)] \
            for t in range(len(y))]
        return np.matrix(trig_form)