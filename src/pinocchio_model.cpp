#include "controllers/dwmpc/pinocchio_model.hpp"

quadrupedModel::quadrupedModel() {}

quadrupedModel::~quadrupedModel() {}

void quadrupedModel::modelInit(parameter const &model_param) {

    std::cout << "quadrupedModel initialization begins..." << std::endl;

    // 参数传递
    subsystems_name_list_ = model_param.subsystems_name;

    n_joint_wb_ = model_param.n_joint_wb;

    n_joint_ = model_param.n_joint;

    n_contact_wb_ = model_param.n_contact_wb;

    n_contact_ = model_param.n_contact;

    N_ = model_param.N_;

    contact_frame_name_list_wb_ = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"};

    J_linear_.resize(n_contact_wb_);

    // 设置文件路径
    std::string urdf_filename{"/usr/include/dls2/controllers/dwmpc/urdf/go1.urdf"};

    // 加载模型
    pinocchio::urdf::buildModel(urdf_filename, pinocchio::JointModelFreeFlyer(), pin_model_);

    // 绑定data和model
    pin_data_ = pinocchio::Data(pin_model_);
    // pinocchio::Data data(pin_model_);
    // pin_data_ = data;


    //组建离散模型矩阵，只修改不变的部分
    for(auto subsystems_name : subsystems_name_list_) {
        if (subsystems_name == "wb") {
            continue;
        } 
        Ak_[subsystems_name] = Eigen::MatrixXd::Identity(37, 37);
        Bk_[subsystems_name] = Eigen::MatrixXd::Zero(37, 6);
    }

    std::cout << "quadrupedModel initialized!!!" << std::endl;

    return;
}      

void quadrupedModel::modelUpdate(std::map<std::string,std::vector<double>> const &xk) {        
    
    Eigen::VectorXd q(19);
    Eigen::VectorXd v(18);
    q.segment(0, 3) = Eigen::Map<const Eigen::VectorXd>(xk.at("p").data(), xk.at("p").size());
    q.segment(3, 4) = Eigen::Map<const Eigen::VectorXd>(xk.at("quat").data(), xk.at("quat").size());
    v.segment(0, 3) = Eigen::Map<const Eigen::VectorXd>(xk.at("dp").data(), xk.at("dp").size());
    v.segment(3, 3) = Eigen::Map<const Eigen::VectorXd>(xk.at("omega").data(), xk.at("omega").size());    
    for (auto i{0};i < n_joint_wb_;i++) {
        q(7+i) = xk.at("q")[i];
        v(6+i) = xk.at("dq")[i];
    }
    
    // 计算所有动力学项
    pinocchio::computeAllTerms(pin_model_, pin_data_, q, v);

    // 从计算结果中提取惯性矩阵、科里奥利力矩阵和重力向量
    Eigen::MatrixXd const &M_wb = pin_data_.M;     // 惯性矩阵
    Eigen::VectorXd const &nle_wb = pin_data_.nle; //包含科里奥利力和重力项

    //计算角速度旋转矩阵
    // Eigen::Quaterniond quat(xk["quat"][3], xk["quat"][0], xk["quat"][1], xk["quat"][2]);
    // Eigen::Vector3d rpy = quatToRPY(quat);
    // Eigen::MatrixXd inv_jac_R = pinocchio::computeRpyJacobianInverse(rpy);
    // Eigen::MatrixXd inv_jac_R = pinocchio::computeRpyJacobianInverse(xk["rpy"]);
    Eigen::MatrixXd inv_jac_R = Eigen::MatrixXd::Identity(3, 3);

    //计算雅可比矩阵，用于计算外部力矩和填充模型参数
    for (size_t i = 0; i < n_contact_wb_; ++i) {
        int frame_id = pin_model_.getFrameId(contact_frame_name_list_wb_[i]);
        Eigen::MatrixXd J(6, pin_model_.nv);
        pinocchio::getFrameJacobian(pin_model_, pin_data_, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J);
        J_linear_[i] = J.topRows(3);
    }

    // 更新子系统
    for(auto subsystems_name : subsystems_name_list_) {
        if (subsystems_name == "wb") {
            continue;
        } 
        updateSubsystem(subsystems_name, pin_data_.M, pin_data_.nle, inv_jac_R, xk);
    }

    return;
}

void quadrupedModel::updateSubsystem(std::string &subsystems_name, Eigen::MatrixXd const &M_wb, 
                                     Eigen::VectorXd const &nle_wb, Eigen::MatrixXd const &inv_jac_R,
                                     std::map<std::string,std::vector<double>> const &xk) {
    int s_idx = 0;
    if (subsystems_name == "front") {
        s_idx = 0;
    } else if (subsystems_name == "back") {
        s_idx = 2;
    } else {
        return;
    }

    //适配子系统的MCG
    Eigen::MatrixXd M(6+n_joint_, 6+n_joint_);
    Eigen::VectorXd nle(6+n_joint_);

    M.block(0, 0, 6, 6) = M_wb.block(0, 0, 6, 6);  // floating base
    M.block(6, 6, n_joint_, n_joint_) = M_wb.block(6+3*s_idx, 6+3*s_idx, n_joint_, n_joint_);
    M.block(0, 6, 6, n_joint_) = M_wb.block(0, 6+3*s_idx, 6, n_joint_);
    M.block(6, 0, n_joint_, 6) = M_wb.block(6+3*s_idx, 0, n_joint_, 6);

    nle.segment(0, 6) = nle_wb.segment(0, 6);
    nle.segment(6, 6) = nle_wb.segment(6+3*s_idx, n_joint_);

    //计算外部力矩，注意是关节力矩不是足端力
    Eigen::VectorXd ext_torque = Eigen::VectorXd::Zero(6+n_joint_);
    Eigen::VectorXd grf = Eigen::Map<const Eigen::VectorXd>(xk.at("grf").data(), xk.at("grf").size());
    for(int idx = 0; idx < n_contact_wb_; ++idx) {
        if ((idx == s_idx) || (idx == (n_contact_-1)+s_idx)) {
            Eigen::VectorXd torque_wb = xk.at("contact")[idx]*J_linear_[idx].transpose()*grf.segment(3*idx, 3);
            ext_torque.segment(0, 6) += torque_wb.segment(0, 6);
            ext_torque.segment(6, 6) += torque_wb.segment(6+3*s_idx, n_joint_);
        } else {
            ext_torque.segment(0, 6) += (xk.at("contact")[idx]*J_linear_[idx].transpose()*grf.segment(3*idx, 3)).segment(0, 6);
        }
    }

    //计算矩阵参数
    Eigen::MatrixXd inv_M = M.inverse();
    Eigen::VectorXd delta = inv_M*(-nle-ext_torque);

    //组建离散模型矩阵，只修改变化的部分
    int dt = 0.02; //dt==loop_dt 或者 dt>loop_dt
    Ak_[subsystems_name].block(0, 12, 3, 3) = Eigen::MatrixXd::Identity(3, 3)*dt;
    Ak_[subsystems_name].block(3, 15, 3, 3) = inv_jac_R*dt;
    Ak_[subsystems_name].block(6, 18, 6, 6) = Eigen::MatrixXd::Identity(6, 6)*dt;
    Ak_[subsystems_name].block(12, 36, 12, 1) = delta*dt;

    Eigen::MatrixXd J_linear_subsystem1(3,12);
    Eigen::MatrixXd J_linear_subsystem2(3,12);
    J_linear_subsystem1.block(0, 0, 3, 6) = J_linear_[s_idx].block(0, 0, 3, 6);
    J_linear_subsystem1.block(0, 6, 3, 6) = J_linear_[s_idx].block(0, 6+3*s_idx, 3, 6);
    J_linear_subsystem2.block(0, 0, 3, 6) = J_linear_[s_idx+1].block(0, 0, 3, 6);
    J_linear_subsystem2.block(0, 6, 3, 6) = J_linear_[s_idx+1].block(0, 6+3*s_idx, 3, 6);

    // if (s_idx == 0) {
    //     J_linear_subsystem1.block(0, 0, 3, 6) = J_linear_[s_idx].block(0, 0, 3, 6);
    //     J_linear_subsystem1.block(0, 6, 3, 6) = J_linear_[s_idx].block(0, 6, 3, 6);
    //     J_linear_subsystem2.block(0, 0, 3, 6) = J_linear_[s_idx+1].block(0, 0, 3, 6);
    //     J_linear_subsystem2.block(0, 6, 3, 6) = J_linear_[s_idx+1].block(0, 6, 3, 6);
    // } else {
    //     J_linear_subsystem1.block(0, 0, 3, 6) = J_linear_[s_idx].block(0, 0, 3, 6);
    //     J_linear_subsystem1.block(0, 6, 3, 6) = J_linear_[s_idx].block(0, 12, 3, 6);
    //     J_linear_subsystem2.block(0, 0, 3, 6) = J_linear_[s_idx+1].block(0, 0, 3, 6);
    //     J_linear_subsystem2.block(0, 6, 3, 6) = J_linear_[s_idx+1].block(0, 12, 3, 6);
    // }

    Ak_[subsystems_name].block(24, 12, 3, 12) = J_linear_subsystem1*dt;
    Ak_[subsystems_name].block(27, 12, 3, 12) = J_linear_subsystem2*dt;
    Ak_[subsystems_name].block(30, 36, 6, 1) = delta.segment(0, 6)*dt;
    Ak_[subsystems_name](36,36) = 1.0;

    Bk_[subsystems_name].block(12, 0, 12, 6) = inv_M.block(0, 6, 12, 6)*dt;
    Bk_[subsystems_name].block(30, 0, 6, 6) = inv_M.block(0, 6, 6, 6)*dt;

    return;
}

std::vector<std::vector<double>> quadrupedModel::updatePrediction(std::vector<double> const &x0,
                                                                  std::vector<std::vector<double>> const &u,
                                                                  std::string const &subsystems_name) {

    std::vector<std::vector<double>> xtraj(N_, std::vector<double>(37, 0.0));
    if (subsystems_name == "wb") {
        return xtraj;
    }
    Eigen::VectorXd xk = Eigen::VectorXd::Map(x0.data(), x0.size());

    // xtraj.push_back(x0);                                 
    for(int i=0; i<N_; ++i) {
        Eigen::VectorXd uk = Eigen::Map<const Eigen::VectorXd>(u[i].data(), u[i].size());
        xk = Ak_[subsystems_name]*xk + Bk_[subsystems_name]*uk;
        xk(3) = normalizeAngle(xk(3));
        xk(4) = normalizeAngle(xk(4));
        xk(5) = normalizeAngle(xk(5));
        // xtraj.push_back(std::vector<double>(xk.data(), xk.data() + xk.size()));
        xtraj[i] = std::vector<double>(xk.data(), xk.data() + xk.size());
    }
                                        
    return xtraj;
}

// 将角度归一化到 -π 到 π 之间
double normalizeAngle(double angle) {
    const double PI = M_PI;
    const double TWO_PI = 2.0 * PI;
    
    // 处理大于2π或小于-2π的角度，先取模
    angle = std::fmod(angle, TWO_PI);
    
    // 如果角度小于 -π，循环增加2π直到在范围内
    if (angle < -PI) {
        angle += TWO_PI;
    }
    // 如果角度大于 π，循环减少2π直到在范围内
    else if (angle > PI) {
        angle -= TWO_PI;
    }
    
    return angle;
}


// std::vector<Eigen::VectorXd> quadrupedModel::updatePrediction(Eigen::VectorXd const &x0,
//                                                               std::vector<Eigen::VectorXd> const &u,
//                                                               std::string const &subsystems_name) {
//     Eigen::VectorXd xk = x0;
//     std::vector<Eigen::VectorXd> xtraj;
//     // xtraj.push_back(x0);                                 
//     for(int i=0; i<N_; ++i) {
//        xk = Ak_[subsystems_name]*xk + Bk_[subsystems_name]*u[i];
//        xtraj.push_back(xk);
//     }
                                        
//     return xtraj;
// }
