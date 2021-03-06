import EspaceState as EE
import numpy as np

class Seasonal:

    def __init__(self):
        self.EE = EE.EspaceState()
        self.__init_exatc()
        self.__init_params()
        self.__init_matrix()
        self.EE.set_system(self.z,self.t,self.r,self.h,self.q,self.c,self.d,self.params)
        self.EE.set_init(self.burn,self.Pinf,self.Pstar)

    def filter(self,y):
        return self.EE.diffuse_filter(y)

    def exact_filter(self,y):
        return self.EE.exact_filter(y)       

    def simulate(self,nr_params):
        return self.EE.simulate(nr_params)

    def ll(self,y):
        return self.EE.ll_func(y)

    def fit(self,y):
        return self.EE.optimize(y)

    def smooth(self,y):
        return self.EE.diffuse_smooth(y)

    def __init_matrix(self):
        self.z = np.matrix([1,0,0,0,0])
        self.t = np.matrix([[-1,-1,-1,-1,-1],[0,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]])
        self.r = np.matrix([[1],[0],[0],[0],[0]])
        self.q = np.matrix([1])
        self.h = np.matrix([1])
        self.c = np.matrix([[0],[0],[0],[0],[0]])
        self.d = np.matrix([[0],[0],[0],[0],[0]])

    def __init_params(self):
        params = {}
        params['Z'] = []
        params['T'] = []
        params['R'] = []
        params['Q'] = []
        params['Q'].append(
            {'position':[0,0],'type':'positive','value':0}
        )
        params['H'] = []
        params['H'].append(
            {'position':[0,0],'type':'positive','value':0}
        )
        params['C'] = []
        params['D'] = []
        self.params = params
    
    def __init_exatc(self):
        self.burn = 5
        self.Pinf = np.matrix([1])
        self.Pstar = np.matrix([0])