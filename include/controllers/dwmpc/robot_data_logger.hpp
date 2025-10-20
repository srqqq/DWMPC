#ifndef ROBOT_DATA_LOGGER_H
#define ROBOT_DATA_LOGGER_H

#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <chrono>
#include <stdexcept>
#include <vector>
#include <map>

class RobotDataLogger {

public:
    // 构造函数，打开文件并准备写入
    RobotDataLogger();
    
    // 析构函数，关闭文件
    ~RobotDataLogger();
    
    // 禁止复制构造和赋值操作
    RobotDataLogger(const RobotDataLogger&) = delete;
    RobotDataLogger& operator=(const RobotDataLogger&) = delete;
    
    // 移动构造和赋值操作
    RobotDataLogger(RobotDataLogger&&) = default;
    RobotDataLogger& operator=(RobotDataLogger&&) = default;
    
    // 记录数据到CSV文件
    bool logData(std::map<std::string, Eigen::VectorXd> const &x, 
        std::map<std::string, std::vector<Eigen::VectorXd>> const &x_ref, 
        std::map<std::string, std::vector<Eigen::VectorXd>> const &u, 
        std::map<std::string, std::vector<Eigen::VectorXd>> const &u_ref,
        std::vector<double> const &residual_l2_norm_time,
        double const &solver_time_wb);

    // 新增的初始化函数
    void init(std::string const &filename);

private:
    std::ofstream file_;
    bool is_first_write_;
    std::string filename_;
    std::chrono::time_point<std::chrono::high_resolution_clock> time_start_;
    int nx_;
    int nu_;
    std::vector<std::string> subsystems_name_;
};

#endif // ROBOT_DATA_LOGGER_H
