# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import inv,det
from scipy.optimize import minimize

class EspaceState:

	def set_system(self,Z,T,R,H,Q,C,D,z_func,t_func):
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
		self.z_func = z_func
		self.t_func = t_func

	
	def set_dims(self,alpha_dim,y_dim,burn):
		self.alpha_dim = alpha_dim
		self.y_dim = y_dim
		self.burn = burn

	def set_init(self,init_params,init_a,init_P):
		self.params = init_params
		self.init_a = init_a
		self.init_P = init_P

	def simulate(self,nr_samples):
		"""Simula nr_samples amostras de uma ocorrencia
		com as variancias dadas por params, que deve ser
		um array na foma [eps_vars,eta_vars]"""
		alpha = [0]*nr_samples
		alpha[0] = self.init_a(self.params)
		y = [0]*nr_samples
		for t in range(nr_samples):
			Eps,Eta = self.__get_disturbances(alpha[t])
			y[t] = (self.z_func(alpha[t],self.params) + Eps).item(0)
			if(t!= nr_samples-1):
				alpha[t+1] = self.t_func(alpha[t],self.params)\
			 	+ np.dot(self.get('R'),Eta)
		return y,alpha

	def diffuse_filter(self,y):
		"""Filtro com inicialização Difusa da série \n
		Devolve um dicionário com todas as matrizes calculadas
		ao longo da filtragem"""
		a,P,F,K,L,inov = self.__get_filter_structures(y)
		a[0] = self.init_a(self.params)	
		a[0][0,0] = 1
		P[0] = self.init_P(self.params)	
		for t in range(len(y)):
			Z, H = self.__get_at_matrix(a[t])	
			inov[t] = y[t] - self.z_func(a[t],self.params) - self.get('D')
			F[t] = Z*P[t]*Z.T + H
			#Passo de Correção
			a[t] = a[t] + P[t]*Z.T*inv(F[t])*inov[t]
			P[t] = P[t] - P[t]*Z.T*inv(F[t])*Z*P[t]
			#Passo de atualização
			T,R,Q = self.__get_att_matrix(a[t])
			if(t!= len(y)-1):
				a[t+1] = self.t_func(a[t],self.params) + self.get('C',a[t])
				P[t+1] = T*P[t]*T.T + R*Q*R.T
		return {'level':a,'variance':P,'inovations':inov,'F':F}


	def __get_at_matrix(self,alpha):
		return self.get('Z',alpha), self.get('H',alpha)

	def __get_att_matrix(self,alpha):
		return self.get('T',alpha), self.get('R',alpha)	, self.get('Q',alpha)	

	def get(self,name, alpha = 0):
		"""Retorna matriz do Sistema especificada por name"""
		Mat = self.Mats[name]
		Mat = Mat(alpha,self.params).astype(float)
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

	def __get_disturbances(self,alpha):
		H = self.get('H',alpha)	
		H_mean = np.zeros(H.shape[0])
		Q = self.get('Q',alpha)	
		Q_mean = np.zeros(Q.shape[0])
		eps_shocks =np.random.multivariate_normal(H_mean,H)
		eta_shocks =np.random.multivariate_normal(Q_mean,Q)
		return [np.matrix(eps_shocks).T, np.matrix(eta_shocks).T]

	def __ll(self,y,params_vect):
		self.params = params_vect
		filtering = self.diffuse_filter(y)
		V = filtering['inovations'][self.burn:]		
		F = filtering['F'][self.burn:]
		ll = [(np.log(det(f)) + v.T*inv(f)*v).item(0) for v,f in zip(V,F)] 
		return sum(ll) 

	def __ll_func(self,y):
		return lambda params_vec : self.__ll(y,params_vec)

	def optimize(self,y):
		ll = self.__ll_func(y)
		fun = 10e7	
		x = []
		for i in range(7):
			#try:
			res =  minimize(ll,np.random.normal(0,2,len(self.params)))
			if res.fun < fun:
				x = res.x	
			#except Exception as e:
				#print('uma otimizaçao falhou')
				#pass
		if x != []:
			self.params = x 
		return x 
	