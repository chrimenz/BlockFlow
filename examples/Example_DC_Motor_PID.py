'''
Created on 19.11.2021

@author: mench
'''

import matplotlib.pyplot as plt
import BlockFlow as bf
from Model_Demo import DC_Motor, PID, Step


# create blocks with models first...
# reference
w = bf.Block(bf.Const([1.0]))
# controller input
wy = bf.Merge()
e = bf.Diff()
# pid
pid = bf.Block(PID(kp=100.0,ki=100.0,kd=1.0,umax=24.0,umin=-24.0,dt=0.01),dt=0.01)
# torque
M = bf.Block(Step(t1=0.4,y1=10.0,dt=0.1),dt=0.1)
# plant input  
u = bf.Merge()
# plant
plant = bf.Block(DC_Motor(), solver='rk4')
# plant output
phi = bf.Selector([0])

# create block flow graph...
bfg = bf.BlockFlowGraph()
# ...add blocks...
bfg.add([w,wy,e,pid,M,u,plant,phi])
# ...and connect
bfg.connect([w, phi], wy)
bfg.connect(wy, e)
bfg.connect(e, pid)
bfg.connect([pid, M], u)
bfg.connect(u, plant)
bfg.connect(plant, phi)

# simulate
bfg.run(1.0)

# plot result
t_phi = bfg.t[phi.id]
y_phi = bfg.y[phi.id][:,:,0]
plt.subplot(2,1,1)
plt.plot(t_phi, y_phi)
plt.grid()

t_u = bfg.t[u.id]
y_u = bfg.y[u.id][:,0,0]
plt.subplot(2,1,2)
plt.plot(t_u, y_u)
plt.grid()
plt.show()