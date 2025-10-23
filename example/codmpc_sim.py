from gym_quadruped.quadruped_env import QuadrupedEnv
import pydwmpc
import numpy as np
import time


robot_name = "go2"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = "flat"  # "flat", "stairs", "ramp", "perlin", "random_boxes", "random_pyramids"
robot_feet_geom_names = dict(FR='FR',FL='FL', RR='RR' , RL='RL')
robot_leg_joints = dict(FR=['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', ],
                        FL=['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint', ],
                        RR=['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', ],
                        RL=['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'])

state_observables_names = tuple(QuadrupedEnv.ALL_OBS)  # return all available state observables

sim_frequency = 200.0
env = QuadrupedEnv(robot=robot_name,
                   scene=scene_name,
                   sim_dt = 1.0/sim_frequency,  # Simulation time step [s]
                   ref_base_lin_vel=0.0, # Constant magnitude of reference base linear velocity [m/s]
                   ground_friction_coeff=1.5,  # pass a float for a fixed value
                   base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
                   state_obs_names=state_observables_names,  # Desired quantities in the 'state'
                   )
obs = env.reset(random=False)

env.render()

mpc = pydwmpc.Dwmpc()
mpc.init()
mpc.startWalking()    
tau = pydwmpc.DoubleVector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
des_q = pydwmpc.DoubleVector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
des_dq = pydwmpc.DoubleVector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
contact = pydwmpc.DoubleVector([0.0,0.0,0.0,0.0])

Kp = 20
Kd = 3
# mpc.setGaitParam(0.8,0.6,1)
# mpc.setStepHeight(0.1)
# mpc.goHandStand()

mpc_frequency = 100.0
mpc_inerval = 1.0/mpc_frequency

start_time = time.time()
is_run_mpc = False

while True:

    duration = time.time() - start_time
    if duration >= mpc_inerval:
        is_run_mpc = True
        start_time = time.time()

    qpos = env.mjData.qpos
    qvel = env.mjData.qvel # 线速度world系下，角速度local系下

    if is_run_mpc:
        # print("duration = ", duration)
        is_run_mpc = False

        env_feet_pos = env.feet_pos('world')
        env_feet_contact_state = env.feet_contact_state()[0]
        foot_op = np.array([env_feet_pos.FL, env_feet_pos.FR, env_feet_pos.RL, env_feet_pos.RR], order="F")
        contact_op = np.array([float(env_feet_contact_state.FL), float(env_feet_contact_state.FR), float(env_feet_contact_state.RL), float(env_feet_contact_state.RR)])

        # foot_op = np.array([env.feet_pos('world').FL, env.feet_pos('world').FR, env.feet_pos('world').RL, env.feet_pos('world').RR],order="F")
        # contact_op = np.array([float(env.feet_contact_state()[0].FL), float(env.feet_contact_state()[0].FR), float(env.feet_contact_state()[0].RL), float(env.feet_contact_state()[0].RR)])

        quat = np.zeros(4)
        quat[0] = qpos[4]
        quat[1] = qpos[5]
        quat[2] = qpos[6]
        quat[3] = qpos[3]

        ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel() # ref_base_lin_vel和ref_base_ang_vel都是world系，详见函数注释
        p = qpos[:3].copy()
        q = qpos[7:].copy()

        dp = qvel[:3].copy()
        # omega = env.base_configuration[:3,:3]@qvel[3:6] # 把local系下的角速度转到world系下
        omega = qvel[3:6]
        dq = qvel[6:].copy()

        mpc.run(p,
            quat,
            q,
            dp,
            omega,
            dq,
            mpc_inerval,
            contact_op,
            foot_op,
            env.heading_orientation_SO3.transpose()@ref_base_lin_vel, # local系的线速度
            ref_base_ang_vel, # 为啥是world系的角速度？？？
            np.array([0.0, 0.0, 0.0, 1.0]),
            contact,
            tau,
            des_q,
            des_dq)

    action_torque = tau + Kp*(des_q.getList() - qpos[7:]) + Kd*(des_dq.getList() - qvel[6:])
    action = np.zeros(env.mjModel.nu)
    action[env.legs_tau_idx.FL] = action_torque[:3]
    action[env.legs_tau_idx.FR] = action_torque[3:6]
    action[env.legs_tau_idx.RL] = action_torque[6:9]
    action[env.legs_tau_idx.RR] = action_torque[9:]
    state, reward, is_terminated, is_truncated, info = env.step(action=action)
    if is_terminated:
        # print("!!!!! mujoco is is_terminated !!!!! timer = ", timer)
        pass
        # Do some stuff
    env.render()
env.close()