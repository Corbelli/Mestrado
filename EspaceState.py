# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import inv,det
from scipy.optimize import minimize

class EspaceState:

	def set_system(self,Z,T,R,H,Q,C,D,params):
		""""Define as matrizes Z,T e alpha, do modelo \n
		y = Z*alpha + eps \n
		alpha_t+1 = T*alpha_t + R*eta, \n
		devem ser entregues como np.matrix \n
		Não retorna"""
		self.Mats = {}
		self.Mats['Z'] = Z
		self.Mats['T'] = T
		self.Mats['R'] = R
		self.Mats['H'] = H
		self.Mats['Q'] = Q
		self.Mats['C'] = C
		self.Mats['D'] = D
		self.params = params

	def simulate(self,nr_samples):
		"""Simula nr_samples amostras de uma ocorrencia
		com as variancias dadas por params, que deve ser
		um array na forma [eps_vars,eta_vars]"""
		alpha_t = np.matrix(np.zeros((self.Mats['Z'].shape[1],1))) 
		y_t = []
		for i in range(nr_samples):
			Eps,Eta = self.__get_disturbances()
			y_t.append( (np.dot(self.get('Z'),alpha_t) + Eps).item(0))
			alpha_t = np.dot(self.get('T'),alpha_t)\
			 + np.dot(self.get('R'),Eta)
		return y_t

	def set_init(self,d,Pinf,Pstar):
		"""Determina as condições da inicialização exata"""
		self.d = d
		self.Pinf = Pinf
		self.Pstar = Pstar
	
	def __exact_recursion(self,Pinf,Pstar,at,y):
		Finf = self.get('Z')*Pinf*self.get('Z').T
		Fstar = self.get('Z')*Pstar*self.get('Z').T + self.get('H')
		if Finf != 0 :
			F1 = inv(Finf)
			F2 = -inv(Finf)*Fstar*inv(Finf)
			K0 = self.get('T')*Pinf*self.get('Z').T*F1
			K1 = self.get('T')*Pstar*self.get('Z').T*F1 + self.get('T')*Pinf*self.get('Z').T*F2
			L0 = self.get('T') - K0*self.get('Z')
			L1 = - K1*self.get('Z')
			at_1 = self.get('T')*at + K0*(y - self.get('Z')*at - self.get('D'))\
			                            + self.get('C')
			Pinf_1 = self.get('T')*Pinf*L0.T
			Pstar_1 = self.get('T')*Pinf*L1.T + self.get('T')*Pstar*L0.T + \
										self.get('R')*self.get('Q')*self.get('R').T
		else :
			K0 = self.get('T')*Pstar*self.get('Z').T*inv(Fstar)
			L0 = self.get('T') - K0*self.get('Z')
			at_1 = self.get('T')*at + K0*(y - self.get('Z')*at - self.get('D'))\
			                            + self.get('C')
			Pinf_1 = self.get('T')*Pinf*self.get('T').T
			Pstar_1 = self.get('T')*Pstar*L0.T + self.get('R')*self.get('Q')*self.get('R').T
		return Finf,Pinf_1,Pstar_1,at_1


	def diffuse_filter(self,y):
		"""Filtro com inicialização Difusa da série \n
		Devolve um dicionário com todas as matrizes calculadas
		ao longo da filtragem"""
		a,P,F,K,L,inov = self.__get_filter_structures(y)
		a[0] = np.matrix(np.zeros((self.Mats['Z'].shape[1],1)))
		P[0] = np.matrix(np.diag([10e5]*self.Mats['R'].shape[0]))
		H = self.get('H')	
		Q = self.get('Q')
		for t in range(len(y)):
			inov[t] = y[t] - self.get('Z')*a[t] - self.get('D')
			F[t] = self.get('Z')*P[t]*self.get('Z').T + H
			K[t] = self.get('T')*P[t]*self.get('Z').T*inv(F[t])
			L[t] = self.get('T') - K[t]*self.get('Z')
			#Passo de Correção
			a[t] = a[t] + P[t]*self.get('Z').T*inv(F[t])*inov[t]
			P[t] = P[t] - P[t]*self.get('Z').T*inv(F[t])*self.get('Z')*P[t]
			#Passo de atualização
			if(t!= len(y)-1):
				a[t+1] = self.get('T')*a[t]	 + self.get('C')
				P[t+1] = self.get('T')*P[t]*self.get('T').T + \
						 self.get('R')*Q*self.get('R').T
		return {'level':a,'variance':P,'inovations':inov,'F':F,'K':K,'L':L}


	def exact_filter(self,y):
		"""Filtro com inicialização exata da série \n
		Devolve um dicionário com todas as matrizes calculadas
		ao longo da filtragem"""
		a,P,F,K,L,inov = self.__get_filter_structures(y)
		a[0] = np.matrix(np.zeros((self.Mats['Z'].shape[1],1)))
		P[0] = self.Pstar 
		Pinf = self.Pinf
		H = self.get('H')	
		Q = self.get('Q')
		for t in range(self.d):
			F[t],Pinf,P[t+1],a[t+1] = self.__exact_recursion(Pinf,P[t],a[t],y[t])

		for t in range(self.d,len(y)):
			inov[t] = y[t] - self.get('Z')*a[t] - self.get('D')
			F[t] = self.get('Z')*P[t]*self.get('Z').T + H
			K[t] = self.get('T')*P[t]*self.get('Z').T*inv(F[t])
			L[t] = self.get('T') - K[t]*self.get('Z')
			#Passo de Correção
			a[t] = a[t] + P[t]*self.get('Z').T*inv(F[t])*inov[t]
			P[t] = P[t] - P[t]*self.get('Z').T*inv(F[t])*self.get('Z')*P[t]
			#Passo de atualização
			if(t!= len(y)-1):
				a[t+1] = self.get('T')*a[t]	 + self.get('C')
				P[t+1] = self.get('T')*P[t]*self.get('T').T + \
						 self.get('R')*Q*self.get('R').T
		return {'level':a,'variance':P,'inovations':inov,'F':F,'K':K,'L':L}

	def diffuse_smooth(self,y):
		"""Suavisador com inicialização Difusa da série \n
		Devolve um dicionário com todas as matrizes calculadas
		ao longo da filtragem"""
		smooth_dict = self.diffuse_filter(y)
		a,P,F,K,L,inov,r,N,s_a,s_P = self.__get_smooth_structures(smooth_dict,y)
		r[len(y)-1] = np.matrix(np.zeros(self.Mats['Z'].shape[1])).T
		for t in range(len(y)-1,0,-1):
			r[t-1] = self.get('Z').T*inv(F[t])*inov[t] + L[t].T*r[t]
			N[t-1] = self.get('Z').T*inv(F[t])*self.get('Z') \
											  + L[t].T*N[t]*L[t]
			s_a[t]   = a[t] + P[t]*r[t-1]
			s_P[t]   = P[t] - P[t]*N[t-1]*P[t]
		smooth_dict['smth_level'] = s_a
		smooth_dict['smth_var'] = s_P
		return smooth_dict

	def get(self,name):
		"""Retorna matriz do Sistema especificada por name"""
		Mat = self.Mats[name].astype(float)
		for param in self.params[name] :
			if param['type'] == 'positive' :
				pos = param['position']
				Mat[pos[0],pos[1]] = np.exp(param['value']) 
			if param['type'] == 'free' :
				pos = param['position']
				Mat[pos[0],pos[1]] = param['value'] 

		return Mat

	def __get_filter_structures(self,y):
		size = len(y)
		a = [0]*size
		P = [0]*size
		F = [0]*size
		K = [0]*size
		L = [0]*size
		inov = [0]*size
		return a,P,F,K,L,inov

	def __get_smooth_structures(self,filtering,y):
		size = len(y)
		a    = filtering['level']
		P    = filtering['variance']
		F    = filtering['F']
		K    = filtering['K']
		L    = filtering['L']
		inov = filtering['inovations']
		r = [0]*size
		N = [0]*size
		s_a = [0]*size
		s_P = [0]*size
		return a,P,F,K,L,inov,r,N,s_a,s_P

	def __get_disturbances(self):
		H = self.get('H')	
		H_mean = np.zeros(self.Mats['H'].shape[0])
		Q = self.get('Q')	
		Q_mean = np.zeros(self.Mats['Q'].shape[0])
		eps_shocks =np.random.multivariate_normal(H_mean,H)
		eta_shocks =np.random.multivariate_normal(Q_mean,Q)
		return [np.matrix(eps_shocks).T, np.matrix(eta_shocks).T]

	def __ll(self,y,params_vect):
		self.__update_params(params_vect)
		filtering = self.diffuse_filter(y)
		V = filtering['inovations'][self.d:]		
		F = filtering['F'][self.d:]
		ll = [(np.log(det(f)) + v.T*inv(f)*v).item(0) for v,f in zip(V,F)] 
		return sum(ll) 

	def __update_params(self,params_vect):
		index = 0
		for Mat in self.Mats:
			for i in range(len(self.params[Mat])):
				self.params[Mat][i]['value'] = params_vect[index]
				index = index + 1
			
	def __ll_func(self,y):
		return lambda params_vec : self.__ll(y,params_vec)

	def optimize(self,y):
		ll = self.__ll_func(y)
		n_params = sum([len(Mat_params) for Mat_params in self.params.values()])
		fun = 10e7	
		x = []
		for i in range(10):
			try:
				res =  minimize(ll,np.random.normal(0,2,n_params))
				if res.fun < fun:
					x = res.x	
			except Exception as e:
				pass
		self.__update_params(x)
		return x