import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class RobotDataLogger:
    """机器人数据记录器，保存到CSV文件"""
    
    # 将固定名称定义为类属性，方便类方法访问
    reference_name = ['ref_x', 'ref_y' , 'ref_yaw', 'ref_vx', 'ref_vy','ref_yawrate']
    base_foot_state_name = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'vx', 'vy', 'vz', 'rollrate', 'pitchrate', 'yawrate',
                            'pf1x', 'pf1y', 'pf1z', 'pf2x', 'pf2y', 'pf2z', 'pf3x', 'pf3y', 'pf3z', 'pf4x', 'pf4y', 'pf4z']
    joint_state_name = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 
                        'dq1', 'dq2', 'dq3', 'dq4', 'dq5', 'dq6', 'dq7', 'dq8', 'dq9', 'dq10', 'dq11', 'dq12']
    control_name = ['tau1', 'tau2', 'tau3', 'tau4', 'tau5', 'tau6', 'tau7', 'tau8', 'tau9', 'tau10', 'tau11', 'tau12',
                    'grf1x', 'grf1y', 'grf1z', 'grf2x', 'grf2y', 'grf2z', 'grf3x', 'grf3y', 'grf3z', 'grf4x', 'grf4y', 'grf4z']

    data_name = reference_name + base_foot_state_name + joint_state_name + control_name
    
    def __init__(self, file_path):
        """初始化日志记录器"""  

        self.file_path = file_path
        self.file = open(file_path, 'w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)
        headers = ['time'] + self.data_name
        self.writer.writerow(headers)

        self.time_start = time.time()

    def parse_full_data(self, full_data):
        """解析全量数据，筛选出需要保存的数据"""

        p = full_data["front"].p[0].getList() # 3
        rpy = full_data["front"].rpy[0].getList() # 3
        dp =  full_data["front"].dp[0].getList() # 3
        omega = full_data["front"].omega[0].getList() #3
        foot = full_data["wb"].foot[0].getList() # 12

        q = full_data["wb"].q[0].getList() # 12
        dq = full_data["wb"].dq[0].getList() # 12

        tau = full_data["wb"].tau[0].getList() # 12
        grf = full_data["wb"].grf[0].getList() # 12

        data = p + rpy + dp + omega + foot + \
               q + dq + \
               tau + grf

        return data

    def log_data(self, plan_state, full_data):
        """添加一组数据到CSV文件"""
        data = self.parse_full_data(full_data)
        time_stamp = time.time() - self.time_start
        row = [time_stamp] + list(plan_state) + list(data)
        self.writer.writerow(row)
        self.file.flush()
    
    @classmethod  # 改为类方法，方便访问类属性
    def plot_data(cls, file_path):  # 移除self参数，直接接收file_path
        """读取机器人数据CSV文件并绘制图形"""
        try:
            print(f"正在读取数据文件: {file_path}")
            df = pd.read_csv(file_path)
            print(f"数据读取成功，共 {len(df)} 条记录")

            time = df['time'].values
            ref_all = [ref.replace('ref_', '') for ref in cls.reference_name] # 从类属性动态提取基础名称（去掉'ref_'前缀）

            print("Create [Base and Foot] graph...")
            fig, axs = plt.subplots(4, 6, figsize=(20, 30))
            fig.suptitle('Base and Foot')
            axs = axs.flatten()
            for i, name in enumerate(cls.base_foot_state_name):  # 使用实际列名
                axs[i].plot(time, df[name], label=name, linewidth=1.5)
                # 检查是否有对应的参考值
                ref_name = f'ref_{name}' if name in ref_all else None
                if ref_name and ref_name in df.columns:
                    axs[i].plot(time, df[ref_name], label=ref_name, linestyle='--', linewidth=1.5)
                axs[i].set_title(name, fontsize=10)
                axs[i].legend(fontsize=8)
                axs[i].xaxis.set_major_locator(MaxNLocator(4))
                axs[i].yaxis.set_major_locator(MaxNLocator(4))
                axs[i].tick_params(axis='both', which='major', labelsize=7)

            print("Create [Joint] graph...")
            fig, axs = plt.subplots(4, 6, figsize=(20, 30))
            fig.suptitle('Joint')
            axs = axs.flatten()
            for i, name in enumerate(cls.joint_state_name):  # 使用实际列名
                axs[i].plot(time, df[name], label=name, linewidth=1.5)
                axs[i].set_title(name, fontsize=10)
                axs[i].legend(fontsize=8)
                axs[i].xaxis.set_major_locator(MaxNLocator(4))
                axs[i].yaxis.set_major_locator(MaxNLocator(4))
                axs[i].tick_params(axis='both', which='major', labelsize=7)

            print("Create [Control Input] graph...")
            fig, axs = plt.subplots(4, 6, figsize=(20, 30))
            fig.suptitle('Control Input')
            axs = axs.flatten()
            for i, name in enumerate(cls.control_name):  # 使用实际列名
                axs[i].plot(time, df[name], label=name, linewidth=1.5)
                axs[i].set_title(name, fontsize=10)
                axs[i].legend(fontsize=8)
                axs[i].xaxis.set_major_locator(MaxNLocator(4))
                axs[i].yaxis.set_major_locator(MaxNLocator(4))
                axs[i].tick_params(axis='both', which='major', labelsize=7)

            print("图形创建完成，显示图形...")
            plt.show()
        except Exception as e:
            print(f"处理数据时发生错误: {str(e)}")    

    def close(self):
        """关闭文件"""
        if hasattr(self, 'file') and not self.file.closed:
            self.file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __del__(self):
        self.close()
