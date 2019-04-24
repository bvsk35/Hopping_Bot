#!/usr/bin/env python3
from mujoco_py import load_model_from_path, MjSim, MjViewer
import math 
import os
Model = "hoppingbot.xml"
model_loaded = load_model_from_path(Model)
sim = MjSim(model_loaded)
viewer = MjViewer(sim)
t = 0

while True:
    sim.data.ctrl[0] = 0.01 * t
    t += 1
    sim.step()
    viewer.render()
