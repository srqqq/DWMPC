#ifndef ROBOT_DATA_LOGGER_H
#define ROBOT_DATA_LOGGER_H

#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <chrono>

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
    bool logData(Eigen::VectorXd const &x, Eigen::VectorXd const &x_ref, 
                 Eigen::VectorXd const &u, Eigen::VectorXd const &u_ref);

    // 新增的初始化函数
    void init(std::string const &filename, int const &nx, int const &nu);

private:
    std::ofstream file_;
    bool is_first_write_;
    std::string filename_;
    std::chrono::time_point<std::chrono::high_resolution_clock> time_start_;
    int nx_;
    int nu_;
};

#endif // ROBOT_DATA_LOGGER_H
