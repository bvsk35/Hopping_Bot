#!/usr/bin/env python3
from mujoco_py import load_model_from_path, MjSim, MjViewer, MjSimState
import math 
import os
import numpy as np
import mujoco_py
import time
Model = "hopper3.xml"
model_loaded = load_model_from_path(Model)
sim = MjSim(model_loaded)
viewer = MjViewer(sim)
viewer.render()
t = 0
flag = False
# sim.data.qpos[0]
while True:
    viewer.render()
    sim.step()
    # if 0.5 > sim.data.time > 0.4:
    #     sim.data.ctrl[2] = 100
    #     sim.data.ctrl[0] = 100
    #     sim.step()
    #     start = time.time()
    #     viewer.render()

    # elif sim.data.time <= 0.4:
    #     sim.data.ctrl[:] = 0
    #     sim.step()
    #     viewer.render()
    # else:
    #     end  = time.time()
    #     if 0.1 > end-start > 0.001:
    #         sim.data.ctrl[1] = -100
    #         sim.data.ctrl[0] = 100
    #         viewer.render()
    #         print(sim.get_state())
    #     # t += 1
    #     # if t > 100:
    #         # t = -10
    #     sim.data.ctrl[:] = 0
    #     sim.step()
    #     # viewer.render()
    #     # print(0.01 * t, "this is the control")
    #     print(sim.get_state())
    #     # sim.data.qpos[1] = 3.14
    #     # print(sim.)
    #     #print(sim.body.xpos('cart'))
    #     # print(sim.get_body_xpos('cart'))
    #     # print(sim.data.qpos[1])
    print(sim.get_state())
    sim.data.ctrl[1] = 0.1
    # sim.set_state(MjSimState(time=0.0, qpos=np.array([1.25, 0.  , 0.  , 0.  , 0.  , 0.  ]), qvel=np.array([0., 0., 0., 0., 0., 0.]), act=None, udd_state={}))
    # ({'time': 0 ,'qpos': np.zeros(6), 'qvel': np.zeros(6)})