'''
Created on 06.03.2021

@author: mench
'''

from numba import jit
import numpy as np

def eul(f, x, u, h):
    _x = x + f(x,u)*h
    return _x

def eulsp(f, dfdx, dfdu, x, u, h, A, B):
    _x = x + f(x,u)*h
    _tmp = np.eye(np.size(A,0)) + dfdx(_x,u)*h
    _A = np.dot(_tmp, A)
    _B = np.dot(_tmp,B) + dfdu(_x,u)*h
    return _x,_A,_B

def rk4(f,x,u,h):
    
    k1 = f(x,u)
    k2 = f(x + 0.5*h*k1,u)
    k3 = f(x + 0.5*h*k2,u)
    k4 = f(x + h*k3,u)
    _x = x + h/6*(k1 + 2*k2 + 2*k3 + k4)
    
    return _x

#@jit #(nopython=True)
def rk4sp(f, dfdx, dfdu, x, u, h, A, B): 
    
    k1 = f(x,u)
    dk1dx = np.dot(dfdx(x,u),A)
    dk1du = np.dot(dfdx(x,u),B) + dfdu(x,u)

    k2 = f(x+h/2*k1,u)
    dk2dx = np.dot(dfdx(x+h/2*k1,u),A+h/2*dk1dx)
    dk2du = np.dot(dfdx(x+h/2*k1,u),B+h/2*dk1du)+dfdu(x+h/2*k1,u)

    k3 = f(x+h/2*k2,u)
    dk3dx = np.dot(dfdx(x+h/2*k2,u),A+h/2*dk2dx)
    dk3du = np.dot(dfdx(x+h/2*k2,u),B+h/2*dk2du)+dfdu(x+h/2*k2,u)

    k4 = f(x+h*k3,u)
    dk4dx = np.dot(dfdx(x+h*k3,u),A+h*dk3dx)
    dk4du = np.dot(dfdx(x+h*k3,u),B+h*dk3du)+dfdu(x+h*k3,u)

    _x = x + h/6*(k1+2*k2+2*k3+k4) 
    _A = A + h/6*(dk1dx + 2*dk2dx + 2*dk3dx + dk4dx)
    _B = B + h/6*(dk1du + 2*dk2du + 2*dk3du + dk4du)

    return _x , _A, _B
