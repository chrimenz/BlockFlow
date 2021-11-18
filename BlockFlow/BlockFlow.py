'''
Created on 05.03.2021

@author: mench
'''
import numpy as np
import copy
from Integrator import rk4


class Block:
    KEY_MDL = 0
    KEY_MRG = 1
    KEY_SEL = 2
    KEY_IN  = 3
    KEY_OUT = 4
    KEY_BFIN = 5
    KEY_FBOUT = 6
    def __init__(self,mdl,dt=0.0,solver='discrete',log=True):
        
        self.nu = mdl.nu
        self.nx = mdl.nx
        self.ny = mdl.ny
        
        self.u = np.zeros([self.nu,1])
        self.x = np.zeros([self.nx,1])
        self.y = np.zeros([self.ny,1])
        
        self.dt = dt
        self.next_t = 0.0
        
        self.mdl = mdl
        
        self.solver = solver
        self.id = -1
        self.key = Block.KEY_MDL
        
        self.log = log
         
    def step(self, u):
        
        self.u = u

        if self.solver == 'discrete':
            self.x = self.mdl.f(self.x, self.u)
        elif self.solver == 'rk4':
            self.x = rk4(self.mdl.f, self.x, self.u, self.dt)
        self.y = self.mdl.h(self.x, self.u)


class Merge_Mdl:
    def __init__(self):
        self.nu = 0
        self.nx = 0
        self.ny = 0
    def f(self,x,u):
        dx = u*0
        return dx
    def h(self,x,u):
        y = u#.reshape([self.ny,1])
        return y 
    
class Merge(Block):
    def __init__(self, dt=0.0):
        super().__init__(Merge_Mdl(),dt=dt,solver='discrete') 
        self.key = Block.KEY_MRG
    def step(self,u):
        super().step(u)
    def expand(self,n):
        self.nu = n
        self.nx = n
        self.ny = n
        self.mdl.nu = n
        self.mdl.nx = n
        self.mdl.ny = n
        self.u = np.zeros([n,1])
        self.x = np.zeros([n,1])
        self.y = np.zeros([n,1])
        
class Selector_Mdl:
    def __init__(self,sel=[0]):
        self.nx = len(sel)
        self.ny = len(sel)
        self.sel = np.r_[sel]
        self.nu = 0
    def f(self,x,u):
        x = u[self.sel]
        return x
    def h(self,x,u):
        y = u[self.sel]
        return y 
    
class Selector(Block):
    def __init__(self, sel,dt=0.0):
        super().__init__(Selector_Mdl(sel),dt=dt,solver='discrete') 
        self.key = Block.KEY_SEL
    def step(self,u):
        super().step(u)
        #self.y = self.u[self.mdl.sel]
    def expand(self,n):
        self.nu = n
        self.mdl.nu = n
        self.u = np.zeros([n,1])
            
class Input_Mdl:
    def __init__(self,n=1):
        self.nx = n
        self.ny = n
        self.nu = n
    def f(self,x,u):
        x = u
        return x
    def h(self,x,u):
        y = u
        return y 
    
class Input(Block):
    def __init__(self, n, dt=0.0):
        super().__init__(Input_Mdl(n),dt=dt,solver='discrete') 
        self.key = Block.KEY_IN
    def step(self,u):
        super().step(u)

class BlockFlowGraph:
    def __init__(self,dt=0.001):
        
        self.blocks = []
        self.prvblks = []
        self.dt = dt
        self.idx = 0
    
    def add(self, b):

        for _b in b:
            self.blocks.append(_b)
            self.blocks[self.idx].id = self.idx
            self.prvblks.append([-1])
            self.idx += 1
    
    def connect(self,b1,b2):

        if b2.key == Block.KEY_MDL:
            self.prvblks[b2.id][0] = b1.id
        elif b2.key == Block.KEY_MRG:
            self.merge(b1, b2)
        elif b2.key == Block.KEY_SEL:
            self.prvblks[b2.id][0] = b1.id
            b2.expand(b1.ny)
        elif b2.key == Block.KEY_IN:
            self.prvblks[b2.id][0] = b1.id
    
    def merge(self,blks,m):
        _m = self.blocks[m.id]
        self.prvblks[m.id] = [0]*len(blks)
        n = 0
        for i,b in enumerate(blks):
            self.prvblks[m.id][i] = b.id
            n += b.ny
            if b.ny < 1:
                raise ValueError('merge error: previous block len')
        _m.expand(n)
        
    def run(self, T):
    
        dt = self.dt
        N = int(T/dt+1)
        tsim = np.arange(0,T+dt,dt)
        next_dot = 1.0
        
        n = len(self.blocks)
        self.u = [ [] for _ in range(n)]
        self.x = [ [] for _ in range(n)]
        self.y = [ [] for _ in range(n)]
        self.t = [ [] for _ in range(n)]
        
        for b in self.blocks:
            #b.next_t = 0.0
            if b.dt < dt:
                b.dt = dt
            
        for k in range(N):
            
            for i,b in enumerate(self.blocks):
                
                if tsim[k] >= b.next_t:
                    b.next_t = b.next_t + b.dt
                    
                    if self.prvblks[i][0] < 0:
                        u = b.u
                    elif b.key == Block.KEY_MRG:
                        u = []
                        for bb in self.prvblks[i]:
                            _u = self.blocks[bb].y.reshape(-1,1)
                            u.extend(_u)
                        u = np.array(u).reshape(b.nu,1)
                    elif b.key == Block.KEY_SEL:
                        u = self.blocks[self.prvblks[i][0]].y.reshape(np.shape(b.u))
                    else:
                        u = self.blocks[self.prvblks[i][0]].y.reshape(np.shape(b.u))
                        
                    b.step(u)
                
                    if b.log:
                        self.u[i].append(b.u)
                        self.x[i].append(b.x)
                        self.y[i].append(b.y)
                        self.t[i].append(np.array(tsim[k]))
                
            if tsim[k] >= next_dot:
                print('.', end='',flush=True)
                next_dot += 1.0
        
        for i in range(len(self.blocks)):
            self.u[i] = np.array(self.u[i])
            self.x[i]  = np.array(self.x[i])
            self.y[i] = np.array(self.y[i])
            self.t[i] = np.array(self.t[i])
                    
        self.tsim = tsim
        
    def clone(self):
        bg = copy.deepcopy(self)
        return bg
    
    def attach(self, bfg):

        offset = self.idx
        maxid = 0
        for i,b in enumerate(bfg.blocks):
            self.blocks.append(b)#(copy.deepcopy(b))
            self.prvblks.append(bfg.prvblks[i])
            if (b.id > -1):
                b.id += offset
                if(b.id > maxid):
                    maxid = b.id
        for i in range(offset,maxid+1):
            for j in range(len(self.prvblks[i])):
                if self.prvblks[i][j] > -1:
                    self.prvblks[i][j] += offset
        
        self.idx = maxid + 1       
        
        
        

class Const:
    def __init__(self, v=1.0):
        self.nx = len(v)
        self.ny = len(v)
        self.nu = len(v)
        self.v = np.array(v).reshape(self.nx,1)
    def f(self,x,u):
        return self.v
    def h(self,x,u):
        return self.v
        
