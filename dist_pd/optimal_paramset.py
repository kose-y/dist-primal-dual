import tensorflow as tf
import numpy as np
from scipy.optimize import minimize
eps = 0.001
cons_unbdd = ({'type': 'ineq', 'fun': lambda x: x[0]-eps}, 
        {'type': 'ineq', 'fun': lambda x: 1-eps-x[0]}, 
        {'type': 'ineq', 'fun': lambda x: x[1]-eps}, 
        {'type': 'ineq', 'fun': lambda x: 0.5-eps-x[1]})
cons_ybdd = ({'type': 'ineq', 'fun': lambda x: x[0]-eps}, 
        {'type': 'ineq', 'fun': lambda x: 1-eps-x[0]}, 
        {'type': 'ineq', 'fun': lambda x: x[1]-eps}, 
        {'type': 'ineq', 'fun': lambda x: 1-eps-x[1]})
cons_bdd = cons_ybdd
cons_stocbdd = ({'type': 'ineq', 'fun': lambda x: x[0]-eps}, 
        {'type': 'ineq', 'fun': lambda x: 1-eps-x[0]}, 
        {'type': 'ineq', 'fun': lambda x: x[1]-eps}, 
        {'type': 'ineq', 'fun': lambda x: 1-eps-x[1]},
        {'type': 'ineq', 'fun': lambda x: x[2]-eps}, 
        {'type': 'ineq', 'fun': lambda x: 1-eps-x[2]},
        {'type': 'ineq', 'fun': lambda x: x[3]-eps}, 
        {'type': 'ineq', 'fun': lambda x: 1-eps-x[3]})

class ParamSet():
    def __init__(self, P, Q, L_F, dnorm):
        self.P = P
        self.Q = Q
        self.L_F = L_F
        self.dnorm = dnorm

    def rho(self, n):
        return 2.0/tf.to_float(n+1)
    def theta(self, n):
        return tf.to_float(n-1)/tf.to_float(n)

    def tau(self, n):
        pass
    def sigma(self, n):
        pass

class UnbddParamSet(ParamSet):
    def __init__(self, N, a,b,c,d,L_F, dnorm, iota=1):
        self.N=N
        self.iota = iota
        def opt_ftn(x):
            q = x[0]
            r = x[1]
            P = 1/(1-q)
            Q = max(a**2/(iota*r*(1-q)), (2*c*d+b**2/q)/(iota*(1-r)), 1/iota)
            return (4*P*L_F/N**2+2*Q*dnorm/N) * (2+ q/(1-q) + (r+0.5)/(0.5-r))
        res = minimize(opt_ftn, [0.24, 0.2], tol=1e-9, constraints=cons_unbdd)
        q, r = res.x

        P = 1.0/(1.0-q)
        Q = max(a**2/(iota*r*(1-q)), (2*c*d+b**2/q)/(iota*(1-r)), 1/iota)

        super().__init__(P, Q, L_F, dnorm)

    def tau(self, n):
        N, P, Q, L_F, dnorm = self.N, self.P, self.Q, self.L_F, self.dnorm
        return tf.to_float(n)/(2*P*L_F+Q*N*dnorm)
    def sigma(self, n):
        N, P, Q, L_F, dnorm, iota = self.N, self.P, self.Q, self.L_F, self.dnorm, self.iota
        return tf.to_float(n)/(iota * N*dnorm) 


class BddParamSet(ParamSet):
    def __init__(self, N, a,b,c,d, L_F, dnorm, omega_x, omega_y):
        self.N=N
        self.omega_x = omega_x
        self.omega_y = omega_y
        def opt_ftn(x):
            q = x[0]
            r = x[1]
            P = 1/(1-q)
            Q = max(a**2/((1-q)*r), 1/(1-r)*(2*c*d+b**2/q))    
            return 4*P*self.omega_x**2/(self.N*(self.N-1))*L_F + 2*omega_x*omega_y*(Q+1)/N*dnorm     
        res = minimize(opt_ftn, [0.24, 0.2], tol=1e-9, constraints=cons_bdd)
        q, r = res.x
        print(q,r)
        P = 1/(1-q)
        Q = max(a**2/((1-q)*r), 1/(1-r)*(2*c*d+b**2/q))
        super().__init__(P, Q, L_F, dnorm)

    def tau(self, n):
        P, Q, L_F, dnorm = self.P, self.Q, self.L_F, self.dnorm
        omega_x, omega_y = self.omega_x, self.omega_y
        return tf.to_float(n)/(2*P*L_F + tf.to_float(n)*Q*dnorm*omega_y/omega_x)

    def sigma(self, n):
        P, Q, L_F, dnorm = self.P, self.Q, self.L_F, self.dnorm
        omega_x, omega_y = self.omega_x, self.omega_y
        return omega_y/(dnorm*omega_x)    

class BddStocParamSet(ParamSet):
    def __init__(self, N, a,b,c,d, L_F, dnorm, Omega_X, Omega_Y, sigma_x, sigma_y):
        self.N, self.Omega_X, self.Omega_Y, self.sigma_x, self.sigma_y = N, Omega_X, Omega_Y, sigma_x, sigma_y
        def opt_ftn(x):
            q,r,s,t = x
            P = 1/(s-q)
            Q = max(a**2/((s-q)*r), (2*c*d+b**2/q)/(t-r))
            return 8*P*L_F*Omega_X**2/(N*(N-1)) + 4*dnorm*Omega_X*Omega_Y*(Q+1)/N +(4*sigma_x*Omega_X+4*sigma_y*Omega_Y)/np.sqrt(N-1) + (2-r)*Omega_X*sigma_x/(3*(1-r)*np.sqrt(float(N-1))) + (2-s)*Omega_Y*sigma_y/(3*(1-s)*np.sqrt(float(N-1)))
        res = minimize(opt_ftn, [0.24, 0.2, 0.33, 0.14], tol=1e-9, constraints=cons_stocbdd)
        q, r, s, t = res.x
        print(q,r,s,t)
        P = 1/(s-q)
        Q = max(a**2/((s-q)*r), (2*c*d+b**2/q)/(t-r))

        super().__init__(P, Q, L_F, dnorm)
     
    def tau(self, n):
        P, Q, N = self.P, self.Q, self.N
        dnorm = self.dnorm
        Omega_X, Omega_Y, L_F, sigma_x = self.Omega_X, self.Omega_Y, self.L_F, self.sigma_x
        return Omega_X*tf.to_float(n)/(2*P*L_F *Omega_X+Q*dnorm*Omega_Y*(N-1)+sigma_x * N* np.sqrt(N-1))

    def sigma(self, n):
        P, Q, N = self.P, self.Q, self.N
        dnorm = self.dnorm
        Omega_X, Omega_Y, L_F, sigma_y = self.Omega_X, self.Omega_Y, self.L_F, self.sigma_y
        return Omega_Y*tf.to_float(n)/(dnorm*Omega_X*(N-1)+sigma_y*N*np.sqrt(N-1))
       
class UnbddStocParamSet(ParamSet):
    def __init__(self, N, a,b,c,d, L_F, dnorm, sigma):
        self.N = N
        #assert a==1
        # Optimization of parameters
        def opt_ftn(x):
            q, r, s, t = x
            P = 1/(s-q)
            Q = max(np.sqrt(a**2/((s-q)*r)), np.sqrt((2*c*d+b**2/q)/(t-r)), 1)
            return (4*P*L_F/(N*(N-1)) + 2*Q*dnorm/N + 2*sigma/np.sqrt(N-1))
        res = minimize(opt_ftn, [0.24, 0.2, 0.33, 0.14], tol=1e-9, constraints=cons_stocbdd)
        q, r, s, t = res.x
        print(q,r,s,t)
        
         
        P = 1/(s-q)
        Q = max(a**2/((s-q)*r), (2*c*d+b**2/q)/(t-r), 1)
  
        super().__init__(P, Q, L_F, dnorm)
     
        self.denom = 2*P*L_F + (N-1)* Q * dnorm +N * tf.sqrt(float(N-1)) *sigma
        self.sig = sigma
    
    def tau(self, n):
        return tf.to_float(n)/tf.to_float(self.denom)

    def sigma(self, n):
        return tf.to_float(n)/(self.dnorm*float(self.N-1) + self.N * tf.sqrt(float(self.N-1))*self.sig)

