#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jamesguo
"""
import numpy as np 
import numpy.random as random
from scipy.optimize import minimize_scalar


class MFG:
    def __init__(self,P,state,action,mu):
        self.nstate = len(state)
        self.naction = len(action)
        self.pi = np.zeros((self.nstate,self.naction))
        self.P = P 
        self.mean_field = mu
        self.horizon = 30
        self.eta = 1
    def reward(self,s,a,mu):
        return -10*(s==1)-1*(a==0)
    
    def transition(self,mu):
        P = np.zeros((self.nstate,self.naction,self.nstate))
        P[1,:,0] = 0.3
        P[1,:,1] = 0.7
        P[0,0,1] = 0.1*mu[1]
        P[0,0,0] = 1-0.1*mu[1]
        P[0,1,1] = 0.75*mu[1]
        P[0,1,0] = 1- 0.75*mu[1]
        return P 
        
    def Population_update(self,pi):
        #pi is a mixed policy here 
        ans = np.zeros((self.nstate))
        #TODO: obtain the final value of ans 
        for s in range(self.nstate):
            for s1 in range(self.nstate):
                for a in range(self.naction):
                    ans[s] += self.mean_field[s1]*pi[s1,a]*self.P[s1,a,s]
        self.mean_field = ans 
        return ans 
    def policy_distance(self,pi,pi_bar):
        diff = np.abs(pi-pi_bar)
        diff = np.sum(diff,axis = 0)
        return np.max(diff)
    def q_distance_inf(self,q1,q2):
        return np.max(np.abs(q1-q2))
    def q_distance_2(self,q1,q2):
        q =q1-q2
        q =q*q
        return np.sqrt(q.sum())
    def h_func(self,x):
        return -np.sum(x*np.log(x))
    # this function is strongly convex corresponding to x
    def Qh_func(self,s,a,pi,mu,iter=500):
        s_temp = s
        a_temp = a
        ans = []
        state = np.arange(self.nstate)
        action = np.arange(self.naction)
        P = self.transition(mu)
        for i in range(iter):
            temp = 0 
            for t in range(self.horizon):
                temp += self.reward(s_temp, a_temp, mu)+self.h_func(pi[s,:])
                s_forward = random.choice(state,size=1,p=P[s_temp,a_temp,:])[0]
                a_forward = random.choice(action,size=1,p=pi[s_forward,:])[0]
                s_temp, a_temp = s_forward,a_forward
            ans.append(temp)
        return np.mean(ans) 
    
    def qh_func(self,s,a,pi,mu,iter=500):
        ans = self.reward(s, a, mu)
        P = self.transition(mu)
        for s1 in range(self.nstate):
            for a1 in range(self.naction):
                ans += P[s,a,s1]*pi[s1,a1]*self.Qh_func(s1, a1, pi, mu,iter)
        
                
        return ans 
    def Vh_func(self,s,pi,mu,iter=500):
        lst = [pi[s,a]*self.Qh_func(s, a, pi, mu,iter) for a in range(self.naction)]
        return np.sum(lst)
    def Gamma_q_func(self,pi,mu,iter=500):
        ans = np.zeros((self.nstate,self.naction))
        for s in range(self.nstate):
            for a in range(self.naction):
                ans[s,a] = self.qh_func(s, a, pi, mu,iter)
            
        return ans 
    
    def mirror(self,q,pi,s):
        #mirror descent step
        def func(u):
            res = u*q[s,0]+(1-u)*q[s,1]-u*np.log(u)-(1-u)*np.log(1-u)-1/2/self.eta*((u-pi[s,0])**2+((1-u)-pi[s,1])**2)
            return res
        interval = (0,1)
        ans = minimize_scalar(func, bounds=interval, method='bounded')
        return np.array([ans.x,1-ans.x])

    
    
 
        
      

  
#test the function of the MFG class 


nstate = 2
naction =2
mu0 = np.array([0.9,0.1])
P = np.zeros((2,2,2))
# H = 0 ,S =1
# Y =0 , N =1
P[1,:,0] = 0.3
P[1,:,1] = 0.7
P[0,0,1] = 0.1*mu0[1]
P[0,0,0] = 1-0.1*mu0[1]
P[0,1,1] = 0.75*mu0[1]
P[0,1,0] = 1- 0.75*mu0[1]
action =np.array([0,1])
state = np.array([0,1])
pi = np.array([[0.7,0.3],[0.1,0.9]])

single = MFG(P,state,action,mu0)
print(single.Qh_func(0, 0, pi, mu0))
print(single.qh_func(0, 1, pi, mu0))
print(single.Vh_func(1, pi, mu0))


# single.evolve(pi)
# single.q_distance_2(q1,q2)
q1 = np.zeros((2,3))
q2 = np.zeros((2,3))
q2[1,1] =3
q2[0,0] = 2
q = q1-q2
q=q*q

print(single.mirror(q, pi, 0))

#####test the function of the mirror 
q = single.Gamma_q_func(pi, mu0)
print(single.mirror(q, pi, 1))



