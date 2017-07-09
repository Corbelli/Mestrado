import ExSpaceState as EE
import numpy as np

class LocalLevel:

    def __init__(self):
        self.EE = EE.EspaceState()
        self.__init_exatc()
        self.__init_params()
        self.__init_matrix()
        z_func = lambda alpha,params : alpha
        t_func = lambda alpha,params : alpha
        self.EE.set_system(self.z,self.t,self.r,self.h,self.q,self.c,self.d,self.params,z_func,t_func)
        self.EE.set_init(self.burn,self.Pinf,self.Pstar)

    def filter(self,y):
        return self.EE.diffuse_filter(y)

    def exact_filter(self,y):
        return self.EE.exact_filter(y)       

    def smooth(self,y):
        return self.EE.diffuse_smooth(y)

    def simulate(self,nr_params):
        return self.EE.simulate(nr_params)

    def ll(self,y):
        return self.EE.ll_func(y)

    def fit(self,y):
        return self.EE.optimize(y) 

    def __init_matrix(self):
        self.z = np.matrix([1])
        self.t = np.matrix([1])
        self.r = np.matrix([1])
        self.q = np.matrix([1])
        self.h = np.matrix([1])
        self.c = np.matrix([0])
        self.d = np.matrix([0])

    def __init_params(self):
        params = {}
        params['Z'] = []
        params['Z'].append(
            {'position':[0,0],'value':0,'use':lambda alpha,value : 1}
        )
        params['T'] = []
        params['R'] = []
        params['Q'] = []
        params['H'] = []
        params['Q'].append(
            {'position':[0,0],'value':0.5,'use':lambda alpha,value :np.exp(alpha.item(0)*value)}
        )
        params['H'].append(
            {'position':[0,0],'value':0.5,'use': lambda alpha,value:np.exp(alpha.item(0)*value)}
        )
        params['C'] = []
        params['D'] = []
        self.params = params
    
    def __init_exatc(self):
        self.burn = 1
        self.Pinf = np.matrix([1])
        self.Pstar = np.matrix([0])