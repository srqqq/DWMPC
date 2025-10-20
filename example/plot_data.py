import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

def visualize_robot_data(csv_file_path):
    """
    读取机器人数据CSV文件并绘制指定的图形
    
    参数:
        csv_file_path1, csv_file_path2: CSV文件路径
    """
    try:

        # title
        state_title = ['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 
                       'vx', 'vy', 'vz', 'yawrate', 'pitchrate', 'rollrate', 'dq1', 'dq2', 'dq3', 'dq4', 'dq5', 'dq6', 
                       'pfx1', 'pfy1', 'pfz1', 'pfx2', 'pfy2', 'pfz2', 
                       'vx', 'vy', 'vz', 'rollrate', 'pitchrate', 'yawrate',
                       '1']
        control_title = ['tau1', 'tau2', 'tau3', 'tau4', 'tau5', 'tau6', 
                         'grf1_x1', 'grf1_y1', 'grf1_z1', 'grf1_x2', 'grf1_y2', 'grf1_z2',
                         'grf2_x1', 'grf2_y1', 'grf2_z1', 'grf2_x2', 'grf2_y2', 'grf2_z2']
        subsystems_name = ['front', 'back']

        # 读取CSV文件
        print(f"正在读取数据文件: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        print(f"数据读取成功，共 {len(df)} 条记录")   

        # 提取时间数据
        time = df['Time'].values

        # 创建第一个窗口：当前状态与参考状态对比
        for problem in subsystems_name :
            print("Create state graph...")
            fig, axs = plt.subplots(6, 6, figsize=(20, 24))  # 6行6列共36个子图
            fig.suptitle(problem + ' state', fontsize=20, y=0.99)
            axs = axs.flatten()  # 将子图数组展平以便迭代

            # 绘制前35组（每组1维状态）
            for i in range(35):
                axs[i].plot(time, df[f'{problem}_x_{i}'], label=f'x_{i}', linewidth=1.5)
                axs[i].plot(time, df[f'{problem}_x_ref_{i}'], label=f'x_ref_{i}', linestyle='--', linewidth=1.5)
                # axs[i].set_title(f'state {i}', fontsize=10)
                axs[i].set_title(state_title[i], fontsize=10)
                axs[i].legend(fontsize=8)
                axs[i].xaxis.set_major_locator(MaxNLocator(4))  # 限制x轴刻度数量
                axs[i].yaxis.set_major_locator(MaxNLocator(4))  # 限制y轴刻度数量
                axs[i].tick_params(axis='both', which='major', labelsize=7)

            # 第36组（包含最后2维状态：35和36）
            i = 35
            axs[i].plot(time, df[f'{problem}_x_{i}'], label=f'x_{i}', linewidth=1.5)
            axs[i].plot(time, df[f'{problem}_x_ref_{i}'], label=f'x_ref_{i}', linestyle='--', linewidth=1.5)
            axs[i].plot(time, df[f'{problem}_x_{i+1}'], label=f'x_{i+1}', linewidth=1.5)
            axs[i].plot(time, df[f'{problem}_x_ref_{i+1}'], label=f'x_ref_{i+1}', linestyle='--', linewidth=1.5)
            # axs[i].set_title(f'state {i} 和 {i+1}', fontsize=10)
            axs[i].set_title(f'{state_title[i]} and {state_title[i+1]}', fontsize=10)
            axs[i].legend(fontsize=8)
            axs[i].xaxis.set_major_locator(MaxNLocator(4))
            axs[i].yaxis.set_major_locator(MaxNLocator(4))
            axs[i].tick_params(axis='both', which='major', labelsize=7)

            # plt.tight_layout(rect=[0, 0, 1, 0.98])  # 为suptitle留出空间

            # 创建第二个窗口：控制输入
            print("Create control input graph...")
            fig, axs = plt.subplots(3, 6, figsize=(20, 24))
            fig.suptitle(problem +' control input', fontsize=20, y=0.99)
            axs = axs.flatten()  # 将子图数组展平以便迭代

            # 绘制18组（每组1维状态）
            for i in range(18):
                axs[i].plot(time, df[f'{problem}_u_{i}'], label=f'u_{i}', linewidth=1.5)
                axs[i].plot(time, df[f'{problem}_u_ref_{i}'], label=f'u_ref_{i}', linestyle='--', linewidth=1.5)
                axs[i].set_title(control_title[i], fontsize=10)
                axs[i].legend(fontsize=8)
                axs[i].xaxis.set_major_locator(MaxNLocator(4))  # 限制x轴刻度数量
                axs[i].yaxis.set_major_locator(MaxNLocator(4))  # 限制y轴刻度数量
                axs[i].tick_params(axis='both', which='major', labelsize=7)


        # solver data
        fig, axs = plt.subplots(2, 1, figsize=(20, 24))
        fig.suptitle('solver data', fontsize=20, y=0.99)
        axs = axs.flatten()  # 将子图数组展平以便迭代

        axs[0].plot(time, df[f'front_residual_l2_norm_time'], label=f'front_residual_l2_norm', linewidth=1.5)
        axs[0].plot(time, df[f'back_residual_l2_norm_time'], label=f'back_residual_l2_norm', linestyle='--', linewidth=1.5)
        axs[0].set_title('residual_l2_norm', fontsize=10)
        axs[0].legend(fontsize=8)
        axs[0].tick_params(axis='both', which='major', labelsize=7)

        axs[1].plot(time, df[f'solver_time_wb'], label=f'solver_time_wb', linewidth=1.5)
        axs[1].set_title('solver_time_wb', fontsize=10)
        axs[1].legend(fontsize=8)
        axs[1].tick_params(axis='both', which='major', labelsize=7)        

        print("图形创建完成，显示图形...")
        plt.show()
   
    except Exception as e:
        print(f"处理数据时发生错误: {str(e)}")

if __name__ == "__main__":
    # 读取并可视化机器人数据
    visualize_robot_data('quadruped_data.csv')
