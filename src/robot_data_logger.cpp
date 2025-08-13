#include "controllers/dwmpc/robot_data_logger.hpp"
#include <stdexcept>
#include <iostream>

RobotDataLogger::RobotDataLogger() {}

RobotDataLogger::~RobotDataLogger() {
    if (file_.is_open()) {
        file_.close();
        std::cout << "数据已保存至: " << filename_ << std::endl;
    }
}

void RobotDataLogger::init(std::string const &filename, int const &nx, int const &nu) {

    filename_ = filename;
    nx_ = nx;
    nu_ = nu;
    is_first_write_ = true;

    // 尝试打开文件
    file_.open(filename_, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename_);
    }

    time_start_ = std::chrono::high_resolution_clock::now();

    return;
}

bool RobotDataLogger::logData(Eigen::VectorXd const &x, Eigen::VectorXd const &x_ref, 
                              Eigen::VectorXd const &u, Eigen::VectorXd const &u_ref) {
    // 检查向量维度是否正确
    // if (x.size() != 37 || x_ref.size() != 37 || u.size() != 18) {
    //     std::cerr << "状态或控制输入维度不正确! 期望: 状态37维, 控制输入18维" << std::endl;
    //     return false;
    // }

    // 计算时间
    std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - time_start_;
    double time = duration.count();

    // 第一次写入时添加标题行
    if (is_first_write_) {
        file_ << "Time";
        
        // 当前状态标题
        for (int i = 0; i < nx_; ++i) {
            file_ << ",x_" << i;
        }
        
        // 参考状态标题
        for (int i = 0; i < nx_; ++i) {
            file_ << ",x_ref_" << i;
        }
        
        // 控制输入标题
        for (int i = 0; i < nu_; ++i) {
            file_ << ",u_" << i;
        }

        // 参考输入标题
        for (int i = 0; i < nu_; ++i) {
            file_ << ",u_ref_" << i;
        }
        
        file_ << std::endl;
        is_first_write_ = false;
    }
    
    // 写入时间
    file_ << time;
    
    // 写入当前状态
    for (int i = 0; i < nx_; ++i) {
        file_ << "," << x[i];
    }
    
    // 写入参考状态
    for (int i = 0; i < nx_; ++i) {
        file_ << "," << x_ref[i];
    }
    
    // 写入控制输入
    for (int i = 0; i < nu_; ++i) {
        file_ << "," << u[i];
    }
    
    // 写入参考输入
    for (int i = 0; i < nu_; ++i) {
        file_ << "," << u_ref[i];
    }    
    file_ << std::endl;
    
    // 检查写入是否成功
    if (file_.fail()) {
        std::cerr << "数据写入失败!" << std::endl;
        return false;
    }
    
    return true;
}
