import numpy as np

class CircularTrajectoryPlanner:
    """圆形轨迹生成器类，用于计算和可视化圆形轨迹"""
    
    @staticmethod
    def normalize_angle(angle):
        """将角度归一化到[-π, π]范围（过圈处理）"""
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def __init__(self, x0, y0, yaw0, radius, speed, clockwise=True):
        """
        初始化轨迹参数
        
        参数:
            x0, y0: 机器人初始位置
            yaw0: 初始航向角(弧度)
            radius: 圆轨迹半径
            speed: 线速度大小(恒定)
            clockwise: 是否顺时针运动
        """
        self.x0 = x0
        self.y0 = y0
        self.yaw0 = yaw0
        self.radius = radius
        self.speed = speed
        self.clockwise = clockwise
        
        # 预计算固定参数
        if clockwise:
            # 顺时针运动参数
            self.cx = x0 - radius * np.sin(yaw0)
            self.cy = y0 + radius * np.cos(yaw0)
            self.yawrate = -speed / radius  # 顺时针角速度为负
        else:
            # 逆时针运动参数
            self.cx = x0 + radius * np.sin(yaw0)
            self.cy = y0 - radius * np.cos(yaw0)
            self.yawrate = speed / radius   # 逆时针角速度为正
            
        self.period = 2 * np.pi * radius / speed  # 圆周运动周期(s)
        self.circle_num = 0 # 运动的圈数
        self.last_circle_num = 0 # 上次记录的运动的圈数
    
    def get_plan_state(self, t):
        """
        根据时间戳t计算轨迹状态
        
        参数:
            t: 当前时间戳(s)
            
        返回:
            轨迹状态元组: (x, y, vx, vy, yaw, yawrate)
        """

        # 记录运动圈数
        self.circle_num = t // self.period
        if self.last_circle_num != self.circle_num:
            print('已经行走'+str(self.circle_num)+'圈')
            self.last_circle_num = self.circle_num

        # 时间归一化到一个周期内
        t_normalized = t % self.period
        angle_step = (self.speed * t_normalized) / self.radius  # 角度变化量
        
        # 计算当前航向角
        if self.clockwise:
            current_angle = self.yaw0 - angle_step
        else:
            current_angle = self.yaw0 + angle_step
            
        # 航向角过圈处理
        yaw = self.normalize_angle(current_angle)
        
        # 计算位置
        x = self.cx + self.radius * np.cos(current_angle - np.pi/2)
        y = self.cy + self.radius * np.sin(current_angle - np.pi/2)
        
        # 计算速度分量
        vx = self.speed * np.cos(current_angle)
        vy = self.speed * np.sin(current_angle)
        
        return (x, y, vx, vy, yaw, self.yawrate)
