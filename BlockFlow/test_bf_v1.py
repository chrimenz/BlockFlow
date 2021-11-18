'''
Created on 06.03.2021

@author: mench
'''

import numpy as np
import matplotlib.pyplot as plt
import BlockFlow as bf
from plant.motor import Motor, MotorArm, HallSensor
from plant.magnet import BrakeMagnet
from plant.vehdyn import VehicleDynamics

from sledctrl.controller import PID,Tz,FBL

# Brake System  
us_const = bf.Block(bf.Const([0.0]),0.001,'discrete') 
us_ML = bf.Merge()
mo = bf.Block(Motor(),0.001,'rk4') 
phi = bf.Selector([0])
bm = bf.Block(MotorArm(),0.001,'discrete')
# Brake Magnet
bh = bf.Block(bf.Const([30.0]),0.001,'discrete')
bmh = bf.Merge()
b_ML = bf.Block(BrakeMagnet(),0.001,'discrete')
b = bf.Selector([0])
ML = bf.Selector([1])
# Vehicle Dynamics
sv = bf.Block(VehicleDynamics(),0.001,'rk4')
al = bf.Block(bf.Const([0.2]),0.1,'discrete')
uf = bf.Merge()

# Brake Controller
ub_const = bf.Block(bf.Const([80.0]),0.001,'discrete')   
ub_b = bf.Merge()
us = bf.Block(PID(kp=2.0,ki=1.0,dt=0.01,umin=-20.0,umax=20.0),0.01,'discrete')
z = bf.Block(Tz(),0.001,'discrete')
K = FBL.getK([-16,-15,-14,-13])
fbl_mdl = FBL(K=K,umin=-20.0,umax=20.0,dt=0.001)
us_fbl = bf.Block(fbl_mdl,0.001,'discrete')
ub_b_x = bf.Merge()

# Speed Controller
uv = bf.Block(bf.Const([5.0]),0.01,'discrete')  
uv_v = bf.Merge()
pid_spd = PID(kp=-20.0,ki=-10.0,dt=0.1,umin=0.0,umax=90.0)
ub = bf.Block(pid_spd,0.1,'discrete')
v = bf.Selector([1])



# Distance Controller


# Sensor
phi_s = bf.Block(HallSensor(),0.001,'discrete')

# Observer
bm_s = bf.Block(MotorArm(),0.001,'discrete')

# Output
mux1 = bf.Merge()


bfg = bf.BlockFlowGraph()

# Plant
bfg.add(us)
bfg.add(us_ML)
bfg.add(mo)
bfg.add(phi)
bfg.add(bm)
bfg.add(bh)
bfg.add(bmh)
bfg.add(b_ML)
bfg.add(b)
bfg.add(ML)
bfg.add(sv)
bfg.add(al)
bfg.add(uf)
bfg.add(ub) 
bfg.add(ub_const) 
bfg.add(ub_b)
bfg.add(uv)
bfg.add(v)
bfg.add(uv_v)
bfg.add(z)
bfg.add(us_fbl)
bfg.add(ub_b_x)
# Sensor
bfg.add(phi_s)
bfg.add(bm_s)
bfg.add(mux1)

bfg.connect([us, ML],us_ML)
bfg.connect(us_ML,mo)
bfg.connect(mo,phi)
bfg.connect(phi,bm)
bfg.connect([bm, bh],bmh)
bfg.connect(bmh,b_ML)
bfg.connect(b_ML,b)
bfg.connect(b_ML,ML)
bfg.connect([b,al],uf)
bfg.connect(uf,sv)
bfg.connect(phi,phi_s)
bfg.connect(phi_s,bm_s)
bfg.connect([bm,bm_s], mux1)

bfg.connect(mo,z)

bfg.connect(sv,v)
bfg.connect([uv, v], uv_v)
bfg.connect(uv_v, ub)
#bfg.connect([ub, bm], ub_b)
bfg.connect([ub_const, bm], ub_b)
#bfg.connect(ub_b, us)
bfg.connect([ub_b, mo], ub_b_x)
bfg.connect(ub_b_x, us_fbl)

sv.x[1] = 10

bfg.run(5)

#plt.plot(bfg.x[mo.id][:,:,0])
plt.plot(bfg.y[us_fbl.id][:,:,0])
plt.grid()
plt.show()

print(bfg.y[ub_b_x.id])