import ExSpaceState2 as EE
import numpy as np

class MultMEB:

    def __init__(self):
        self.EE = EE.EspaceState()
        self.__init_matrix()
        self.__set_init()
        self.EE.set_system(self.z,self.t,self.r,self.h,self.q,self.c,self.d,self.z_func,self.t_func)

    def filter(self,y):
        return self.EE.diffuse_filter(y)

    def simulate(self,nr_params):
        return self.EE.simulate(nr_params)

    def ll(self,y):
        return self.EE.ll_func(y)

    def fit(self,y):
        return self.EE.optimize(y) 

    def __init_matrix(self):
        self.z = lambda alpha,params : self.Z(alpha,params)
        self.t = lambda alpha,params : self.T(alpha,params)
        self.r = lambda alpha,params : self.R(alpha,params)
        self.q = lambda alpha,params : self.Q(alpha,params)
        self.h = lambda alpha,params : self.H(alpha,params)
        self.c = lambda alpha,params : self.C(alpha,params)
        self.d = lambda alpha,params : self.D(alpha,params)
        self.z_func = lambda alpha,params : self.Z_func(alpha,params)
        self.t_func = lambda alpha,params : self.T_func(alpha,params)

    def __set_init(self):
        init_params = []
        init_a = np.matrix([4,1]).T
        k = 10e5 
        init_P = np.matrix([[k,0],
                            [0,1]]) 
        self.EE.set_init(init_params,init_a,init_P)
        self.EE.set_dims(alpha_dim =2,y_dim = 1,burn = 1)

    def Z_func(self,alpha,params):
        return np.matrix([alpha.item(0)*alpha.item(1)])
    def T_func(self,alpha,params):
        return self.T(alpha,params)*alpha
    @staticmethod
    def Z(alpha,params):
        return np.matrix([alpha.item(1),alpha.item(0)])
    @staticmethod
    def T(alpha,params):
        return np.matrix([[1,0],
                          [0,0]])
    @staticmethod
    def R(alpha,params):
        return  np.matrix([[0],
                           [1]])
    @staticmethod
    def H(alpha,params):
        return np.matrix([0])
    @staticmethod
    def Q(alpha,params):
        return np.matrix([1])
    @staticmethod
    def C(alpha,params):
        return np.matrix([[0],
                          [0]])
    @staticmethod
    def D(alpha,params):
        return np.matrix([0])



def frac(x):
    return np.exp(x)/(1 + np.exp(x))

def pos(x):
    return np.exp(x)

def freq(x):
    return (np.pi*2)/np.exp(x) 