#include "controllers/dwmpc/robot_data_logger.hpp"

RobotDataLogger::RobotDataLogger() {}

RobotDataLogger::~RobotDataLogger() {
    if (file_.is_open()) {
        file_.close();
        std::cout << "数据已保存至: " << filename_ << std::endl;
    }
}

void RobotDataLogger::init(std::string const &filename) {

    filename_ = filename;
    subsystems_name_ = {"front", "back"};
    nx_ = 37;
    nu_ = 18;
    is_first_write_ = true;

    // 尝试打开文件
    file_.open(filename_, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) {
        throw std::runtime_error("无法打开文件: " + filename_);
    }

    time_start_ = std::chrono::high_resolution_clock::now();

    return;
}

bool RobotDataLogger::logData(std::map<std::string, Eigen::VectorXd> const &x, std::map<std::string, std::vector<Eigen::VectorXd>> const &x_ref, 
                              std::map<std::string, std::vector<Eigen::VectorXd>> const &u, std::map<std::string, std::vector<Eigen::VectorXd>> const &u_ref) {

    // 计算时间
    std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - time_start_;
    double time = duration.count();

    // 第一次写入时添加标题行
    if (is_first_write_) {
        file_ << "Time";
        
        for(auto problem : subsystems_name_) {
            std::string str = "," + problem + "_";

            // 当前状态标题
            for (int i = 0; i < nx_; ++i) {
                file_ << str + "x_" << i;
            }
            
            // 参考状态标题
            for (int i = 0; i < nx_; ++i) {
                file_ <<  str + "x_ref_" << i;
            }
            
            // 控制输入标题
            for (int i = 0; i < nu_; ++i) {
                file_ <<  str + "u_" << i;
            }

            // 参考输入标题
            for (int i = 0; i < nu_; ++i) {
                file_ <<  str + "u_ref_" << i;
            }
        }
        file_ << std::endl;
        is_first_write_ = false;
    }
    
    // 写入时间
    file_ << time; 

    // 写入当前状态
    for(auto problem : subsystems_name_) {

        for (int i = 0; i < nx_; ++i) {
            file_ << "," << x.at(problem)(i);
        }
        
        // 写入参考状态
        for (int i = 0; i < nx_; ++i) {
            file_ << "," << x_ref.at(problem)[0](i);
        }
        
        // 写入控制输入
        for (int i = 0; i < nu_; ++i) {
            file_ << "," << u.at(problem)[0](i);
        }
        
        // 写入参考输入
        for (int i = 0; i < nu_; ++i) {
            file_ << "," << u_ref.at(problem)[0](i);
        }
    }
    file_ << std::endl;
    
    // 检查写入是否成功
    if (file_.fail()) {
        std::cerr << "数据写入失败!" << std::endl;
        return false;
    }
    
    return true;
}
