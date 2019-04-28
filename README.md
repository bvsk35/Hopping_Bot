# Hopping_Bot
Guided Policy Search on Hopping task
GPS implementation on hopping bot.

# Files to add:
- [ ] Algorithm_BADMM.py
- [ ] Trajectory_Opt_LQR.py
- [ ] Algoithm.py
- [ ] Hyperparams.py
- [ ] Cost_Sum.py
- [ ] Cost_State.py
- [ ] Cost_Action.py
- [ ] Agent.py

# TODO:
- [ ] Figure out what is Policy GMM and why are they are using it? Where does the code for Policy go? Where it is called?
- [ ] Recheck the files required for Trajectory Optimization
- [ ] Look how cost functions are written and how they are generalized?
- [ ] Read more about Agent.py file. What is the function of this file? Where it is called?
- [ ] Understand how Traj_Opt_Utils.py computes KL-Divergence.
- [ ] Do we need noise patterns in the LinearGaussianPolicy file?
- [ ] In GitHub add result folder and simulator results.
- [ ] Figure out what these files do: gps.py (main file)
- [ ] Figure out what these files do: PolicyOptCaffe.py (then modify it into tensorflow based policy)
- [ ] Figure out what these files do: gps.gui.config.py (Rendering plus GUI can we make our code without using this?)
- [ ] Figure out what these files do: gps.proto.gps_pb2.py (can we make our code without using this?)
