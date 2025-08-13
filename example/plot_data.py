import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

def visualize_robot_data(csv_file_path):
    """
    读取机器人数据CSV文件并绘制指定的图形
    
    参数:
        csv_file_path: CSV文件路径
    """
    try:
        # 读取CSV文件
        print(f"正在读取数据文件: {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        print(f"数据读取成功，共 {len(df)} 条记录")
        
        # 提取时间数据
        time = df['Time'].values
        
        # 创建第一个窗口：当前状态与参考状态对比
        print("Create state graph...")
        # fig1, axs1 = plt.subplots(9, 4, figsize=(20, 24))  # 9行4列共36个子图
        fig1, axs1 = plt.subplots(6, 6, figsize=(20, 24))  # 6行6列共36个子图
        fig1.suptitle('state', fontsize=20, y=0.99)
        axs1 = axs1.flatten()  # 将子图数组展平以便迭代
        
        # title
        state_title = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 
                       'vx', 'vy', 'vz', 'rollrate', 'pitchrate', 'yawrate', 'dq1', 'dq2', 'dq3', 'dq4', 'dq5', 'dq6', 
                       'pfx1', 'pfy1', 'pfz1', 'pfx2', 'pfy2', 'pfz2', 
                       'vx', 'vy', 'vz', 'rollrate', 'pitchrate', 'yawrate',
                       '1']
        control_title = ['tau1', 'tau2', 'tau3', 'tau4', 'tau5', 'tau6', 
                         'grfx1', 'grfy1', 'grfz1', 'grfx2', 'grfy2', 'grfz2',
                         'grfauxx1', 'grfauxy1', 'grfauxz1', 'grfauxx2', 'grfauxy2', 'grfauxz2']

        # 绘制前35组（每组1维状态）
        for i in range(35):
            axs1[i].plot(time, df[f'x_{i}'], label=f'x_{i}', linewidth=1.5)
            axs1[i].plot(time, df[f'x_ref_{i}'], label=f'x_ref_{i}', linestyle='--', linewidth=1.5)
            # axs1[i].set_title(f'state {i}', fontsize=10)
            axs1[i].set_title(state_title[i], fontsize=10)
            axs1[i].legend(fontsize=8)
            axs1[i].xaxis.set_major_locator(MaxNLocator(4))  # 限制x轴刻度数量
            axs1[i].yaxis.set_major_locator(MaxNLocator(4))  # 限制y轴刻度数量
            axs1[i].tick_params(axis='both', which='major', labelsize=7)
        
        # 第36组（包含最后2维状态：35和36）
        i = 35
        axs1[i].plot(time, df[f'x_{i}'], label=f'x_{i}', linewidth=1.5)
        axs1[i].plot(time, df[f'x_ref_{i}'], label=f'x_ref_{i}', linestyle='--', linewidth=1.5)
        axs1[i].plot(time, df[f'x_{i+1}'], label=f'x_{i+1}', linewidth=1.5)
        axs1[i].plot(time, df[f'x_ref_{i+1}'], label=f'x_ref_{i+1}', linestyle='--', linewidth=1.5)
        # axs1[i].set_title(f'state {i} 和 {i+1}', fontsize=10)
        axs1[i].set_title(f'{state_title[i]} and {state_title[i+1]}', fontsize=10)
        axs1[i].legend(fontsize=8)
        axs1[i].xaxis.set_major_locator(MaxNLocator(4))
        axs1[i].yaxis.set_major_locator(MaxNLocator(4))
        axs1[i].tick_params(axis='both', which='major', labelsize=7)
        
        # plt.tight_layout(rect=[0, 0, 1, 0.98])  # 为suptitle留出空间

        # 创建第二个窗口：控制输入
        print("Create control input graph...")
        fig2, axs2 = plt.subplots(3, 6, figsize=(20, 24))
        fig2.suptitle('control input', fontsize=20, y=0.99)
        axs2 = axs2.flatten()  # 将子图数组展平以便迭代

        # 绘制18组（每组1维状态）
        for i in range(18):
            axs2[i].plot(time, df[f'u_{i}'], label=f'u_{i}', linewidth=1.5)
            axs2[i].plot(time, df[f'u_ref_{i}'], label=f'u_ref_{i}', linestyle='--', linewidth=1.5)
            axs2[i].set_title(control_title[i], fontsize=10)
            axs2[i].legend(fontsize=8)
            axs2[i].xaxis.set_major_locator(MaxNLocator(4))  # 限制x轴刻度数量
            axs2[i].yaxis.set_major_locator(MaxNLocator(4))  # 限制y轴刻度数量
            axs2[i].tick_params(axis='both', which='major', labelsize=7)
        
        print("图形创建完成，显示图形...")
        plt.show()
        
    except Exception as e:
        print(f"处理数据时发生错误: {str(e)}")

if __name__ == "__main__":
    # 读取并可视化机器人数据
    visualize_robot_data('front_data.csv')
