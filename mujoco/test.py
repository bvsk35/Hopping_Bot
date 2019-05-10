#!/usr/bin/env python3
from mujoco_py import load_model_from_path, MjSim, MjViewer
import math 
import os
import mujoco_py
Model = "cartpole.xml"
model_loaded = load_model_from_path(Model)
sim = MjSim(model_loaded)
# viewer = MjViewer(sim)
t = 0
while True:
    sim.data.ctrl[0] = 0.1
    # t += 1
    if t > 100:
        t = -10
    sim.step()
    # viewer.render()
    print(0.01 * t, "this is the control")
    print(sim.get_state())
    # sim.data.qpos[1] = 3.14
    # print(sim.)
    #print(sim.body.xpos('cart'))
    # print(sim.get_body_xpos('cart'))
    # print(sim.data.qpos[1])
