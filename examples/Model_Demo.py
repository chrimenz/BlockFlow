'''
Created on 19.11.2021

@author: mench
'''

import numpy as np

class DC_Motor:
    def __init__(self,**kwargs):
        
        self.nx = 3
        self.nu = 2
        self.ny = 3
        
        self.p = {'km':5.36, 'kr':0.76, 'J':0.276, 'L':0.006, 'R':1.9 }
        self.setPars(**kwargs)
        
    def setPars(self, **kwargs):  
        for key, value in kwargs.items(): 
            if key in self.p:
                self.p[key] = value
                
        km,kr,J,L,R = self.p.values()
        
        self.A = np.array( [[0, 1, 0],
                            [0, -kr/J, km/J ],
                            [0, -km/L, -R/L ] 
                           ])
        self.B = np.array([ [0, 0], 
                            [0, -1/J ], 
                            [1/L, 0 ] 
                          ]) 
        self.C = np.eye(len(self.A))
            
    def f(self,x,u):
        dx = np.dot(self.A, x) + np.dot(self.B, u)
        return dx
    
    def h(self,x,u=0):
        y = np.dot(self.C, x)
        return y
    

class PID:
    def __init__(self,**kwargs):
        self.ei = 0
        self.e  = 0
        self.ed = 0
        self.nx = 3
        self.nu = 1
        self.ny = 1
        self.p = { 'kp':1.0, 'ki':0.0, 'kd':0.0, 'umin':-np.inf, 'umax':np.inf, 'dt':1.0}
        self.setPars(**kwargs)
    def setPars(self,**kwargs):
        for key, value in kwargs.items(): 
            if key in self.p:
                self.p[key] = value
    def f(self, x, u):
        x[0] = self.e
        x[1] = self.ei
        x[2] = self.ed
        return x
    
    def h(self, x, _u):
        
        e = _u[0]
        
        kp = self.p['kp']
        ki = self.p['ki']
        kd = self.p['kd']
        dt = self.p['dt']
        umin = self.p['umin']
        umax = self.p['umax']

        ei = self.ei + e*dt
        ed = (e - self.e)/dt
        
        u = kp*e
        u = u + ki*ei
        u = u + kd*ed
        
        if u >= umin:
            if u <= umax:
                self.ei = ei
            else:
                u = umax
        else:
            u = umin 
        
        self.e = e
        self.ed = ed
        
        _y = np.zeros([1,1])
        _y[0] = u
        
        return _y

class Step:
    def __init__(self,**kwargs):
        self.t = 0.0
        self.p = { 'y0':0.0, 't1':1.0, 'y1':1.0, 'dt':1.0}
        self.setPars(**kwargs)
        self.nu = 1
        self.nx = 1
        self.ny = 1
    def setPars(self,**kwargs):
        for key, value in kwargs.items(): 
            if key in self.p:
                self.p[key] = value
    def f(self,x,u):
        return u
    def h(self,x,u):
        y = self.run()
        return np.array(y)
    def run(self):
        y0,t1,y1,dt = self.p.values()
        u = y0
        if self.t >= t1:
            u = y1
        self.t += dt
        return np.array(u)