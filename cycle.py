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

        init_params =[0,0,0,2,0,-3,0.1]
        init_a = lambda params : np.matrix([1,1,1]).T 
        init_P = lambda params : np.matrix([[10e5,0                                      ,                                      0],
                                            [0   ,pos(params[2])/(1-(frac(params[4])**2)),                                      0],
                                            [0   ,0                                      ,pos(params[2])/(1-(frac(params[4])**2))]]) 
        self.EE.set_init(init_params,init_a,init_P)
        self.EE.set_dims(alpha_dim =3,y_dim = 1,burn = 3)

    

    def Z_func(self,alpha,params):
        mu = alpha.item(0)
        gamma = alpha.item(1)
        c_mu = params[6]
        c_0 = params[5]
        exp = np.exp(c_0 + c_mu*mu)
        return np.matrix([mu + gamma*exp])
    def T_func(self,alpha,params):
        return self.T(alpha,params)*alpha
    @staticmethod
    def Z(alpha,params):
        mu = alpha.item(0)
        gamma = alpha.item(1)
        c_mu = params[6]
        c_0 = params[5]
        exp = np.exp(c_0 + c_mu*mu)
        return np.matrix( [1 + gamma*c_mu*exp,exp, 0])
    @staticmethod
    def T(alpha,params):
        return   np.matrix([[1,                                       0,                                       0],
                            [0, np.cos(freq(params[3]))*frac(params[4]), np.sin(freq(params[3]))*frac(params[4])],
                            [0,-np.sin(freq(params[3]))*frac(params[4]), np.cos(freq(params[3]))*frac(params[4])]])

    @staticmethod
    def R(alpha,params):
        return  np.matrix([[1,0,0],
                           [0,1,0],
                           [0,0,1]])
    @staticmethod
    def H(alpha,params):
        return np.matrix([pos(params[0])])
    @staticmethod
    def Q(alpha,params):
        return np.matrix([[pos(params[1]),0             ,             0],
                          [0             ,pos(params[2]),             0],
                          [0             ,0             ,pos(params[2])]])
    @staticmethod
    def C(alpha,params):
        return np.matrix([[0],
                          [0],
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