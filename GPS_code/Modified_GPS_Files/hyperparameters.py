import numpy as np


SENSOR_DIMS = {
    'JOINT_ANGLES': 4,
    'JOINT_VELOCITIES': 4,
    'TORSO_POSTION': 2,
    'TORSO_VELOCITIES': 2,
    'ACTION': 3,
}

Agent_params = {
    'Time' : 100,
    'No_condition' : 4, 
    'No_of_actuator' : 3,
    'world_path' : ['random.xml'], #TODO: add a mujoco path folder for all the things
    'render' : True,
    'initial_condition':[np.zeros(12),
                        0.1 + np.zeros(12),
                        0.3 + np.zeros(12),
                        0.4 + np.zeros(12)],
    'state_include': ['JOINT_ANGLES', 'JOINT_VELOCITIES', 'TORSO_POSTION',
                      'TORSO_VELOCITIES'],
    'obs_include': ['JOINT_ANGLES', 'TORSO_POSTION']
}

#T is time steps
#Number of rollouts = No_rollouts