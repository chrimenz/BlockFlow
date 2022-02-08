'''
Created on 19.11.2021

@author: mench
'''

import matplotlib.pyplot as plt
import BlockFlow as bf
from Model_Demo import DC_Motor


# create blocks with models first...
# plant input 
u = bf.Block(bf.Const([10.0, 0.0]))
# plant
plant = bf.Block(DC_Motor(), solver = 'rk4')
# plant output
phi = bf.Selector([0])
omega = bf.Selector([1])

# create block flow graph...
bfg = bf.BlockFlowGraph()
# ...add blocks...
bfg.add([u,plant,phi,omega])
# ...and connect
bfg.connect(u, plant)
bfg.connect(plant, phi)
bfg.connect(plant, omega)

# simulate
bfg.run(0.5)
print('done')

# plot result
t_omega = bfg.t[omega.id]
y_omega = bfg.y[omega.id][:,:,0]
plt.plot(t_omega, y_omega)
plt.grid()
plt.show()



