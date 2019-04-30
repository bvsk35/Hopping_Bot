# Hoppingbot
Closed loop control of hopping bot with RL

# Files to add:
- [ ] Algorithm_BADMM.py
- [ ] Trajectory_Opt_LQR.py
- [ ] Algoithm.py
- [ ] Hyperparams.py
- [x] Cost_Sum.py
- [x] Cost_State.py
- [x] Cost_Action.py
- [x] Cost_Utils.py
- [ ] Agent.py

# TODO:
- [x] Add function to calculate nominal trajectories using state dynamic matrices given by GMM.
- [x] Install TensorFlow GPU in both Python 2 and Python 3
- [ ] In GitHub add result folder and simulator results.
- [ ] All the Cost files and remaining files depend on hyperparameter file. Now work with Agent.py and Hyperparams.py file.
- [ ] Recheck the files required for Trajectory Optimization
- [ ] Look how cost functions are written and how they are generalized?
- [ ] Read more about Agent.py file. What is the function of this file? Where it is called?
- [ ] Understand how Traj_Opt_Utils.py computes KL-Divergence.
- [ ] Do we need noise patterns in the LinearGaussianPolicy file?
- [ ] Figure out what is Policy GMM and why are they are using it? Where does the code for Policy go? Where it is called?
- [ ] Figure out what these files do: gps.py (main file)
- [ ] Figure out what these files do: PolicyOptCaffe.py (then modify it into tensorflow based policy)
- [ ] Figure out what these files do: gps.gui.config.py (Rendering plus GUI can we make our code without using this?)
- [ ] Figure out what these files do: gps.proto.gps_pb2.py (can we make our code without using this?)

