from gym_quadruped.quadruped_env import QuadrupedEnv
import pydwmpc
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from trajectory_planner import CircularTrajectoryPlanner
from robot_data_logger import RobotDataLogger

''' 仿真功能设置 '''
scene_name = "flat"  # 仿真场景选择，可选参数："flat", "stairs", "ramp", "perlin", "random_boxes", "random_pyramids"
is_traj_mode_enable = True # True：自动跟随轨迹（按一下上方向键开始）；False：手动控制速度

''' Mujoco设置 '''
robot_name = "go2"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
scene_name = "flat"  # "flat", "stairs", "ramp", "perlin", "random_boxes", "random_pyramids"
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

''' MPC设置 '''
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

mpc_start_time = time.time()
is_run_mpc = False

# is_traj_mode_enable = True # 想手动控制速度需要把它改为False
is_traj_mode_enable = False # 想手动控制速度需要把它改为False
is_traj_start = False
traj_start_time = time.time()
plan_state = [0] * 6

ref_base_lin_vel = np.array([0.0, 0.0, 0.0])
ref_base_ang_vel = np.array([0.0, 0.0, 0.0])

''' 数据记录设置 '''
is_log_enable = True # 是否开启数据记录
robot_data_logger = RobotDataLogger("quadruped_data.csv")

''' 主循环 '''
try:
    while True:

        '''获取Mujoco状态反馈'''
        qpos = env.mjData.qpos
        qvel = env.mjData.qvel # 线速度world系下，角速度local系下
        rotation = R.from_quat([qpos[4], qpos[5], qpos[6], qpos[3]]) # Define a quaternion (x, y, z, w)
        euler_angles = rotation.as_euler('ZYX', degrees=False) # 注意：旋转顺序大小写字母表达的意思不同！！！大写表示转轴！！！

        ''' 轨迹生成部分 '''
        if is_traj_mode_enable and not is_traj_start and ref_base_lin_vel[0] >= 0.01:
            is_traj_start = True
            traj_planner = CircularTrajectoryPlanner(
                x0=qpos[0],           # 初始X位置
                y0=qpos[1],           # 初始Y位置
                yaw0=euler_angles[0], # 初始航向角
                radius=1.0,           # 轨迹半径
                speed=0.2,            # 线速度
                clockwise=False)      # 逆时针运动
            traj_start_time = time.time()

        if is_traj_start:
            ''' 自动跟踪轨迹 '''
            traj_duration = time.time() - traj_start_time
            plan_state = traj_planner.get_plan_state(traj_duration)
            ref_base_lin_vel = np.array([plan_state[3], plan_state[4], 0.0])
            ref_base_ang_vel = np.array([0.0, 0.0, plan_state[5]])
        else:
            ''' 手动控制速度 '''
            ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel() # ref_base_lin_vel和ref_base_ang_vel都是world系，详见函数注释
            plan_state = [qpos[0], qpos[1], euler_angles[0], ref_base_lin_vel[0], ref_base_lin_vel[1], ref_base_ang_vel[2]]

        ''' MPC定时调度 '''
        mpc_duration = time.time() - mpc_start_time
        if mpc_duration >= mpc_inerval:
            is_run_mpc = True
            mpc_start_time = time.time()

        if is_run_mpc:
            # print("mpc_duration = ", mpc_duration)
            is_run_mpc = False

            env_feet_pos = env.feet_pos('world')
            env_feet_contact_state = env.feet_contact_state()[0]
            foot_op = np.array([env_feet_pos.FL, env_feet_pos.FR, env_feet_pos.RL, env_feet_pos.RR], order="F")
            contact_op = np.array([float(env_feet_contact_state.FL), float(env_feet_contact_state.FR), float(env_feet_contact_state.RL), float(env_feet_contact_state.RR)])
            # foot_op = np.array([env.feet_pos('world').FL, env.feet_pos('world').FR, env.feet_pos('world').RL, env.feet_pos('world').RR],order="F")
            # contact_op = np.array([float(env.feet_contact_state()[0].FL), float(env.feet_contact_state()[0].FR), float(env.feet_contact_state()[0].RL), float(env.feet_contact_state()[0].RR)])

            quat = np.zeros(4) 
            quat[0] = qpos[4] # x
            quat[1] = qpos[5] # y
            quat[2] = qpos[6] # z
            quat[3] = qpos[3] # w

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
                # mpc_inerval,
                mpc_duration,
                contact_op,
                foot_op,
                env.heading_orientation_SO3.transpose()@ref_base_lin_vel, # local系的线速度
                ref_base_ang_vel, # 为啥是world系的角速度？？？
                np.array([0.0, 0.0, 0.0, 1.0]),
                contact,
                tau,
                des_q,
                des_dq)
            
            if is_log_enable:
                full_data = mpc.getFullPrediction()
                robot_data_logger.log_data(plan_state, full_data)

        ''' 底层关节控制器 '''
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
except KeyboardInterrupt:
    print("用户中断程序")
finally:
    print("程序停止")
    env.close()
