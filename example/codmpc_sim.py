from gym_quadruped.quadruped_env import QuadrupedEnv
import pydwmpc
import numpy as np
import sys
from pathlib import Path

from disturbance import sample_sinusoidal_lateral_force
from trajectory_planner import create_trajectory_planner, normalize_angle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_tools import MCAPDataLogger  # noqa: E402

''' 仿真功能开关 '''
scene_name = "perlin"  # 仿真场景选择，可选参数："flat", "stairs", "ramp", "perlin", "slippery", "random_boxes", "random_pyramids"
is_log_enable = True # True：开启数据记录 False：关闭数据记录

''' 轨迹模式与参数（在此处统一设置） '''
TRAJECTORY_MODE_MANUAL = "manual"
TRAJECTORY_MODE_CIRCLE = "circle"
TRAJECTORY_MODE_FIGURE_EIGHT = "figure_eight"
TRAJECTORY_MODES = (
    TRAJECTORY_MODE_MANUAL,
    TRAJECTORY_MODE_CIRCLE,
    TRAJECTORY_MODE_FIGURE_EIGHT,
)

trajectory_mode = TRAJECTORY_MODE_FIGURE_EIGHT
trajectory_start_speed_threshold = 0.01
trajectory_tracking_kp_position = 1.5
trajectory_tracking_kp_yaw = 1.0
trajectory_parameters = {
    TRAJECTORY_MODE_CIRCLE: {
        "radius": 1.0,
        "speed": 0.2,
    },
    TRAJECTORY_MODE_FIGURE_EIGHT: {
        "x_amplitude": 1.0,  # 轨迹局部坐标系x方向幅值 [m]
        "y_amplitude": 0.5,  # 轨迹局部坐标系y方向幅值 [m]
        "period": 30.0,      # 完成一个8字的时间 [s]
    },
}
if trajectory_mode not in TRAJECTORY_MODES:
    raise ValueError(f"unknown trajectory mode: {trajectory_mode}")
trajectory_is_automatic = trajectory_mode != TRAJECTORY_MODE_MANUAL

''' 扰动实验参数（相对轨迹开始时间，方向为参考航向坐标系+y） '''
disturbance_parameters = {
    "enabled": True,
    "start_delay": 1.0,       # 轨迹开始后等待时间 [s]
    "amplitude_ratio": 0.1,   # 峰值力 / 机器人重量
    "frequency": 1.0,         # [Hz]，当前等于MPC步态频率
    "phase": 0.0,             # [rad]
    "ramp_cycles": 1.0,       # 起止平滑过渡各占用的周期数
    "steady_cycles": 1000.0,  # 保持完整幅值的周期数
}

''' Mujoco设置 '''
robot_name = "go2"   # "aliengo", "mini_cheetah", "go2", "hyqreal", ...
state_observables_names = tuple(QuadrupedEnv.ALL_OBS)  # return all available state observables

sim_frequency = 200.0
ground_friction_coeff = 1.0
env = QuadrupedEnv(robot=robot_name,
                   scene=scene_name,
                   sim_dt = 1.0/sim_frequency,  # Simulation time step [s]
                   ref_base_lin_vel=0.0, # Constant magnitude of reference base linear velocity [m/s]
                   ground_friction_coeff=ground_friction_coeff,  # pass a float for a fixed value
                   base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
                   state_obs_names=state_observables_names,  # Desired quantities in the 'state'
                   )
obs = env.reset(random=False)
env.render()

base_body_id = int(env.mjModel.jnt_bodyid[0])
robot_weight = (
    env.mjModel.body_subtreemass[base_body_id]
    * np.linalg.norm(env.mjModel.opt.gravity)
)
if disturbance_parameters["enabled"]:
    print(
        "正弦侧向扰动: "
        f"A={disturbance_parameters['amplitude_ratio']:.2f}mg "
        f"({disturbance_parameters['amplitude_ratio'] * robot_weight:.2f} N), "
        f"f={disturbance_parameters['frequency']:.2f} Hz"
    )

''' MPC设置 '''
mpc = pydwmpc.Dwmpc()
mpc.init()
mpc.startWalking()    
tau = np.zeros(12)
des_q = np.zeros(12)
des_dq = np.zeros(12)

Kp = 20
Kd = 3
# mpc.setGaitParam(0.8,0.6,1)
# mpc.setStepHeight(0.1)

mpc_frequency = 100.0
mpc_interval = 1.0/mpc_frequency
mpc_interval_steps = int(round(sim_frequency/mpc_frequency))
if not np.isclose(mpc_interval_steps/sim_frequency, mpc_interval):
    raise ValueError("sim_frequency must be an integer multiple of mpc_frequency")
sim_step = 0

''' 轨迹设置 '''
is_traj_start = False
traj_start_time = env.mjData.time
plan_state = np.zeros(6)
ref_base_lin_vel = np.array([0.0, 0.0, 0.0])
ref_base_ang_vel = np.array([0.0, 0.0, 0.0])

''' 数据记录设置 '''
robot_data_logger = None
if is_log_enable:
    friction_label = f"mu{ground_friction_coeff:.1f}".replace(".", "p")
    condition_parts = [scene_name, friction_label]
    if disturbance_parameters["enabled"]:
        amplitude_label = str(
            disturbance_parameters["amplitude_ratio"]
        ).replace(".", "p")
        frequency_label = str(
            disturbance_parameters["frequency"]
        ).replace(".", "p")
        condition_parts.append(f"sin_y_a{amplitude_label}mg_f{frequency_label}hz")
    robot_data_logger = MCAPDataLogger(
        PROJECT_ROOT / "data" / "experiment",
        vehicle=robot_name,
        trajectory=trajectory_mode,
        method="codmpc",
        condition="+".join(condition_parts),
    )
    print(f"MCAP日志将保存到: {robot_data_logger.file_path}")


def build_mpc_snapshot(snapshot_time, p, quat, rpy, q, dp, omega, dq,
                       feedback_foot_position_world, feedback_contact_state,
                       feedback_grf_world, reference_plan_state,
                       reference_command_linear_velocity_world,
                       reference_command_angular_velocity_world, mpc_result,
                       disturbance, tau_pd, tau_cmd, action, qfrc_actuator):
    """Build one independent, flat snapshot at the pre-step control boundary."""
    mpc_node = mpc_result.snapshot
    plan_state_array = np.asarray(reference_plan_state, dtype=float)
    return {
        't': float(snapshot_time),
        'mpc_success': float(mpc_result.success),
        'feedback_base_position_world': np.asarray(p, dtype=float).copy(),
        'feedback_base_quaternion_xyzw': np.asarray(quat, dtype=float).copy(),
        'feedback_base_rpy': np.asarray(rpy, dtype=float).copy(),
        'feedback_joint_position': np.asarray(q, dtype=float).copy(),
        'feedback_base_linear_velocity_world': np.asarray(dp, dtype=float).copy(),
        'feedback_base_angular_velocity_world': np.asarray(omega, dtype=float).copy(),
        'feedback_joint_velocity': np.asarray(dq, dtype=float).copy(),
        'feedback_foot_position_world': np.asarray(
            feedback_foot_position_world, dtype=float
        ).copy(),
        'feedback_contact_state': np.asarray(
            feedback_contact_state, dtype=float
        ).copy(),
        'feedback_grf_world': np.asarray(feedback_grf_world, dtype=float).copy(),
        'feedback_generalized_actuator_force': np.asarray(
            qfrc_actuator, dtype=float
        ).copy(),
        'reference_position_xy_world': plan_state_array[0:2].copy(),
        'reference_yaw': float(plan_state_array[2]),
        'reference_feedforward_linear_velocity_world': np.array(
            [plan_state_array[3], plan_state_array[4], 0.0]
        ),
        'reference_feedforward_angular_velocity_world': np.array(
            [0.0, 0.0, plan_state_array[5]]
        ),
        'reference_command_linear_velocity_world': np.asarray(
            reference_command_linear_velocity_world, dtype=float
        ).copy(),
        'reference_command_angular_velocity_world': np.asarray(
            reference_command_angular_velocity_world, dtype=float
        ).copy(),
        'reference_contact_state': np.asarray(mpc_result.contact, dtype=float).copy(),
        'disturbance_active': float(disturbance.active),
        'disturbance_elapsed_time': float(disturbance.elapsed_time),
        'disturbance_force_reference_heading': np.asarray(
            disturbance.force_reference_heading, dtype=float
        ).copy(),
        'disturbance_force_world': np.asarray(
            disturbance.force_world, dtype=float
        ).copy(),
        'prediction_base_position_world': np.asarray(
            mpc_node.position, dtype=float
        ).copy(),
        'prediction_base_rpy': np.asarray(mpc_node.rpy, dtype=float).copy(),
        'prediction_base_linear_velocity_world': np.asarray(
            mpc_node.linear_velocity, dtype=float
        ).copy(),
        'prediction_base_rpy_rate': np.asarray(
            mpc_node.rpy_rate, dtype=float
        ).copy(),
        'prediction_foot_position_world': np.asarray(
            mpc_node.foot_position, dtype=float
        ).copy(),
        'prediction_joint_position': np.asarray(
            mpc_node.joint_position, dtype=float
        ).copy(),
        'prediction_joint_velocity': np.asarray(
            mpc_node.joint_velocity, dtype=float
        ).copy(),
        'control_mpc_joint_torque': np.asarray(
            mpc_result.torque, dtype=float
        ).copy(),
        'control_pd_joint_torque': np.asarray(tau_pd, dtype=float).copy(),
        'control_joint_torque': np.asarray(tau_cmd, dtype=float).copy(),
        'control_actuator_command': np.asarray(action, dtype=float).copy(),
        'control_mpc_grf_world': np.asarray(
            mpc_node.ground_reaction_force, dtype=float
        ).copy(),
    }

''' 主循环 '''
try:
    while True:

        '''获取Mujoco状态反馈'''
        qpos = env.mjData.qpos
        qvel = env.mjData.qvel # 线速度world系下，角速度local系下
        # rotation = R.from_quat([qpos[4], qpos[5], qpos[6], qpos[3]]) # Define a quaternion (x, y, z, w)
        # euler_angles = rotation.as_euler('ZYX', degrees=False) # 注意：旋转顺序大小写字母表达的意思不同！！！大写表示转轴！！！
        euler_angles = env.base_ori_euler_xyz

        ''' 轨迹生成部分 '''
        if (trajectory_is_automatic
                and not is_traj_start
                and np.linalg.norm(ref_base_lin_vel[:2])
                >= trajectory_start_speed_threshold):
            is_traj_start = True
            traj_planner = create_trajectory_planner(
                trajectory_mode,
                x0=qpos[0],
                y0=qpos[1],
                yaw0=euler_angles[2],
                parameters=trajectory_parameters[trajectory_mode],
            )
            traj_start_time = env.mjData.time

        if is_traj_start:
            ''' 自动跟踪轨迹 '''
            traj_duration = env.mjData.time - traj_start_time
            plan_state = traj_planner.get_plan_state(traj_duration)
            velocity_xy = (
                plan_state[3:5]
                + trajectory_tracking_kp_position * (plan_state[:2] - qpos[:2])
            )
            yaw_rate = (
                plan_state[5]
                + trajectory_tracking_kp_yaw
                * normalize_angle(plan_state[2] - euler_angles[2])
            )
            ref_base_lin_vel = np.array([velocity_xy[0], velocity_xy[1], 0.0])
            ref_base_ang_vel = np.array([0.0, 0.0, yaw_rate])
        else:
            ''' 手动控制速度 '''
            ref_base_lin_vel, ref_base_ang_vel = env.target_base_vel(frame='world')
            plan_state = np.array([
                qpos[0], qpos[1], euler_angles[2],
                ref_base_lin_vel[0], ref_base_lin_vel[1], ref_base_ang_vel[2],
            ])

        trajectory_time = (
            env.mjData.time - traj_start_time if is_traj_start else 0.0
        )
        disturbance = sample_sinusoidal_lateral_force(
            trajectory_time,
            plan_state[2],
            robot_weight,
            enabled=disturbance_parameters["enabled"] and is_traj_start,
            start_delay=disturbance_parameters["start_delay"],
            amplitude_ratio=disturbance_parameters["amplitude_ratio"],
            frequency=disturbance_parameters["frequency"],
            phase=disturbance_parameters["phase"],
            ramp_cycles=disturbance_parameters["ramp_cycles"],
            steady_cycles=disturbance_parameters["steady_cycles"],
        )
        env.mjData.xfrc_applied[base_body_id] = 0.0
        env.mjData.xfrc_applied[base_body_id, :3] = disturbance.force_world

        ''' MPC定时调度：使用仿真步数，不使用墙钟时间 '''
        is_run_mpc = sim_step % mpc_interval_steps == 0
        if is_run_mpc:

            env_feet_pos = env.feet_pos('world')
            env_feet_contact_state, _, env_grf_world = env.feet_contact_state(
                frame='world', ground_reaction_forces=True
            )
            foot_op = np.array([env_feet_pos.FL, env_feet_pos.FR, env_feet_pos.RL, env_feet_pos.RR], order="F")
            contact_op = np.array([float(env_feet_contact_state.FL), float(env_feet_contact_state.FR), float(env_feet_contact_state.RL), float(env_feet_contact_state.RR)])
            grf_measured_world = np.concatenate([
                env_grf_world.FL,
                env_grf_world.FR,
                env_grf_world.RL,
                env_grf_world.RR,
            ])

            quat = np.zeros(4) 
            quat[0] = qpos[4] # x
            quat[1] = qpos[5] # y
            quat[2] = qpos[6] # z
            quat[3] = qpos[3] # w

            p = qpos[:3].copy()
            q = qpos[7:].copy()

            # MuJoCo free-joint linear velocity is world-frame, while its angular
            # velocity is body-frame.  Dwmpc::run expects both in the world frame.
            dp = qvel[:3].copy()
            omega = env.base_configuration[:3, :3] @ qvel[3:6]
            dq = qvel[6:].copy()

            mpc_result = mpc.run(p,
                quat,
                q,
                dp,
                omega,
                dq,
                mpc_interval,
                contact_op,
                foot_op,
                ref_base_lin_vel,
                ref_base_ang_vel)

            tau = np.asarray(mpc_result.torque)
            des_q = np.asarray(mpc_result.joint_position)
            des_dq = np.asarray(mpc_result.joint_velocity)
            mpc_success = mpc_result.success

        ''' 底层关节控制器 '''
        tau_mpc = tau
        tau_pd = Kp*(des_q - qpos[7:]) + Kd*(des_dq - qvel[6:])
        tau_cmd = tau_mpc + tau_pd
        action = np.zeros(env.mjModel.nu)
        action[env.legs_tau_idx.FL] = tau_cmd[:3]
        action[env.legs_tau_idx.FR] = tau_cmd[3:6]
        action[env.legs_tau_idx.RL] = tau_cmd[6:9]
        action[env.legs_tau_idx.RR] = tau_cmd[9:]
        ctrl_limited = np.asarray(env.mjModel.actuator_ctrllimited, dtype=bool)
        ctrl_range = np.asarray(env.mjModel.actuator_ctrlrange)
        action = np.clip(action,
                         np.where(ctrl_limited, ctrl_range[:, 0], -np.inf),
                         np.where(ctrl_limited, ctrl_range[:, 1], np.inf))

        if is_run_mpc and is_log_enable:
            snapshot = build_mpc_snapshot(
                env.mjData.time,
                p,
                quat,
                euler_angles,
                q,
                dp,
                omega,
                dq,
                foot_op,
                contact_op,
                grf_measured_world,
                plan_state,
                ref_base_lin_vel,
                ref_base_ang_vel,
                mpc_result,
                disturbance,
                tau_pd,
                tau_cmd,
                action,
                env.mjData.qfrc_actuator,
            )
            robot_data_logger.log(snapshot)

        state, reward, is_terminated, is_truncated, info = env.step(action=action)

        sim_step += 1
        env.render()
except KeyboardInterrupt:
    print("用户中断程序")
finally:
    print("程序停止")
    if robot_data_logger is not None:
        robot_data_logger.close()
    env.close()
