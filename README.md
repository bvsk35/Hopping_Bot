# Hoppingbot
Implementation of Guided Policy Search (GPS) on Hopping Bot

# How to Install
- Update: `sudo apt-get update`
- Jupyter Notebook: Run `python3 -m pip install --upgrade pip` and then `python3 -m pip install jupyter`
- Required Dependencies: Run `pip3 install -r requirement.txt`
- iLQR module: Go to ilqr-master_new and run `python3 setup.py install`
- GPU Drivers: To install drivers for Nvidia which are needed for running GPU follow the instructions in the following [link](https://www.tensorflow.org/install/gpu). Check if drivers are connected and they are responding by running `nvidia-smi`.
- Update and reboot (not necessary but recommened): `sudo apt-get update && sudo reboot`
- Finally run `TrajGenerator-V2.ipynb`

# Files to add:
- [ ] Algorithm_BADMM.py
- [ ] Trajectory_Opt_LQR.py
- [ ] Algoithm.py
- [ ] Hyperparams.py
- [ ] Agent.py
- [ ] Agent_Utils.py
- [ ] Sample.py
- [ ] Sample_List.py
- [ ] Agent_Config.py
- [x] Cost_Sum.py
- [x] Cost_State.py
- [x] Cost_Action.py
- [x] Cost_Utils.py
- [x] Config_ALG_BADMM.py
- [x] Config_Traj_Opt.py
- [x] Algorithm_Utils.py
- [x] Traj_Opt_Utils.py
- [x] Linear_Gauss_Policy.py
- [x] GPS_General_Utils.py
- [x] Policy.py

# TODO:
- [x] Test MUJOCO Cart Pole and write function to extract states, send control and to control rendering. 
- [x] Figure out what does the following function do `class BundleType`.
- [x] Figure out what does the following function do `func extract_condition`.
- [x] Add function to calculate nominal trajectories using state dynamic matrices given by GMM.
- [x] Install TensorFlow GPU in both Python 2 and Python 3.
- [x] In GitHub add result folder and simulator results.
- [ ] All the Cost files and remaining files depend on hyperparameter file. Now work with Agent.py and Hyperparams.py file.
- [x] Recheck the files required for Trajectory Optimization.
- [x] Add functio to calculate the nominal states and actions (i.e. iLQR optimized) using state dynamic matrices given by GMM. 
- [ ] Look how cost functions are written and how they are generalized?
- [ ] Read more about Agent.py file. What is the function of this file? Where it is called?
- [ ] Understand how Traj_Opt_Utils.py computes KL-Divergence.
- [ ] Do we need noise patterns in the LinearGaussianPolicy file?
- [ ] Figure out what is Policy GMM and why are they are using it? Where does the code for Policy go? Where it is called?
- [ ] Figure out what these files do: `gps.py` (main file).
- [ ] Figure out what these files do: `PolicyOptCaffe.py` (then modify it into tensorflow based policy).
- [ ] Figure out what these files do: `gps.gui.config.py` (Rendering plus GUI can we make our code without using this?).
- [ ] Figure out what these files do: `gps.proto.gps_pb2.py` (can we make our code without using this?).

