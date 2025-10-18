#include "controllers/dwmpc/pinocchio_model.hpp"

quadrupedModel::quadrupedModel() {}

quadrupedModel::~quadrupedModel() {}

void quadrupedModel::modelInit(parameter const &config_param) {

    std::cout << "quadrupedModel initialization begins..." << std::endl;

    // 参数传递
    subsystems_name_list_ = config_param.subsystems_name;

    config_param_ = config_param;

    contact_frame_name_list_wb_ = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"};

    joints_name_list_wb_ = {"FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"};

    J_linear_wb_.resize(config_param_.n_contact_wb);
    J_linear_sub_.resize(config_param_.n_contact_wb);
    J_linear_leg_.resize(config_param_.n_contact_wb);
    // J_linear_submix_.resize(config_param_.n_contact_wb);

    // world_J_linear_wb_.resize(config_param_.n_contact_wb);
    // world_J_linear_sub_.resize(config_param_.n_contact_wb);
    // world_J_linear_leg_.resize(config_param_.n_contact_wb);
    // world_J_linear_submix_.resize(config_param_.n_contact_wb);

    // 设置文件路径
    std::string urdf_filename{"/usr/include/dls2/controllers/dwmpc/urdf/go2.urdf"};

    using joint_pair_t = std::pair<const std::string, std::shared_ptr<::urdf::Joint>>;

    ::urdf::ModelInterfaceSharedPtr urdfTree = ::urdf::parseURDFFile(urdf_filename);
    if (urdfTree == nullptr) {
        throw std::invalid_argument("The file " + urdf_filename + " does not contain a valid URDF model!");
    }

    // remove extraneous joints from urdf
    ::urdf::ModelInterfaceSharedPtr newModel = std::make_shared<::urdf::ModelInterface>(*urdfTree);
    for (joint_pair_t& jointPair : newModel->joints_) {
        if (std::find(joints_name_list_wb_.begin(), joints_name_list_wb_.end(), jointPair.first) == joints_name_list_wb_.end()) {
            jointPair.second->type = urdf::Joint::FIXED;
        }
    }

    // add 6 DoF for the floating base
    pinocchio::JointModelComposite jointComposite(2);
    jointComposite.addJoint(pinocchio::JointModelTranslation());
    jointComposite.addJoint(pinocchio::JointModelSphericalZYX());
    // 加载模型
    pinocchio::urdf::buildModel(urdfTree, jointComposite, pin_model_);

    // 加载模型
    // pinocchio::urdf::buildModel(urdf_filename, pinocchio::JointModelFreeFlyer(), pin_model_);

    // 绑定data和model
    pin_data_ = pinocchio::Data(pin_model_);

    //组建离散模型矩阵，只修改不变的部分
    for(auto subsystems_name : subsystems_name_list_) {
        if (subsystems_name == "wb") {
            continue;
        } 
        Ak_[subsystems_name] = Eigen::MatrixXd::Zero(config_param_.n_state, config_param_.n_state);
        Bk_[subsystems_name] = Eigen::MatrixXd::Zero(config_param_.n_state, config_param_.n_control);
    }

    std::cout << "quadrupedModel initialized!!!" << std::endl;

    return;
}      

void quadrupedModel::modelUpdate(std::map<std::string,std::vector<double>> const &x0_map) {        
    
    Eigen::VectorXd q(18);
    Eigen::VectorXd v(18);

    q.segment(0, 3) = Eigen::Map<const Eigen::VectorXd>(x0_map.at("p").data(), x0_map.at("p").size());
    // q.segment(3, 4) = Eigen::Map<const Eigen::VectorXd>(x0_map.at("quat").data(), x0_map.at("quat").size());
    q(3) = x0_map.at("rpy")[2];
    q(4) = x0_map.at("rpy")[1];
    q(5) = x0_map.at("rpy")[0];
    v.segment(0, 3) = Eigen::Map<const Eigen::VectorXd>(x0_map.at("dp").data(), x0_map.at("dp").size());
    // v.segment(3, 3) = Eigen::Map<const Eigen::VectorXd>(x0_map.at("omega").data(), x0_map.at("omega").size());    
    v(3) = x0_map.at("omega")[2];
    v(4) = x0_map.at("omega")[1];
    v(5) = x0_map.at("omega")[0];
    for (auto i{0};i < config_param_.n_joint_wb;i++) {
        q(6+i) = x0_map.at("q")[i];
        v(6+i) = x0_map.at("dq")[i];
    }

    // 计算所有动力学项
    // pinocchio::computeAllTerms(pin_model_, pin_data_, q, v);

    // // pinocchio只计算了M的上三角部分，需要填充M下三角部分!!!
    // pin_data_.M.triangularView<Eigen::StrictlyLower>() = pin_data_.M.transpose().triangularView<Eigen::StrictlyLower>();

    // 参考qiayuan的调用方法
    pinocchio::forwardKinematics(pin_model_, pin_data_, q, v);
    pinocchio::updateFramePlacements(pin_model_, pin_data_);
    pinocchio::computeJointJacobians(pin_model_, pin_data_);
    pinocchio::crba(pin_model_, pin_data_, q);
    // pinocchio只计算了M的上三角部分，需要填充M下三角部分!!!
    pin_data_.M.triangularView<Eigen::StrictlyLower>() = pin_data_.M.transpose().triangularView<Eigen::StrictlyLower>();
    pinocchio::nonLinearEffects(pin_model_, pin_data_, q, v);

    // 从计算结果中提取惯性矩阵、科里奥利力矩阵和重力向量
    Eigen::MatrixXd const &M_wb = pin_data_.M;     // 惯性矩阵
    Eigen::VectorXd const &nle_wb = pin_data_.nle; //包含科里奥利力和重力项

    //计算角速度旋转矩阵
    // Eigen::Quaterniond quat(x0_map["quat"][3], x0_map["quat"][0], x0_map["quat"][1], x0_map["quat"][2]);
    // Eigen::Vector3d rpy = quatToRPY(quat);
    // Eigen::MatrixXd inv_jac_R = pinocchio::computeRpyJacobianInverse(rpy);
    // Eigen::MatrixXd inv_jac_R = pinocchio::computeRpyJacobianInverse(x0_map["rpy"]);
    Eigen::MatrixXd inv_jac_R = Eigen::MatrixXd::Identity(3, 3);

    //计算雅可比矩阵，用于计算外部力矩和填充模型参数
    for (size_t i = 0; i < config_param_.n_contact_wb; ++i) {
        int frame_id = pin_model_.getFrameId(contact_frame_name_list_wb_[i]);

        // pinocchio::LOCAL_WORLD_ALIGNED
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, pin_model_.nv);
        pinocchio::getFrameJacobian(pin_model_, pin_data_, frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J);
        //wb
        J_linear_wb_[i] = J.topRows(3);
        //leg
        J_linear_leg_[i] = J_linear_wb_[i].block(0, 6+i*3, 3, 3);
        //sub       
        int s_idx = (i < config_param_.n_contact? 0 : 2);
        Eigen::MatrixXd J_temp = Eigen::MatrixXd::Zero(3, 12);
        J_temp.block(0, 0, 3, 6) = J_linear_wb_[i].block(0, 0, 3, 6);
        J_temp.block(0, 6, 3, 6) = J_linear_wb_[i].block(0, 6+3*s_idx, 3, 6);
        J_linear_sub_[i] = J_temp;
        //submix
        // J_temp = Eigen::MatrixXd::Zero(3, 12);
        // J_temp.block(0, 0, 3, 6) = J_linear_wb_[i].block(0, 0, 3, 6);
        // int start_col = ( (i==0 || i==2)? 6 : 9 );
        // J_temp.block(0, start_col, 3, 3) = J_linear_wb_[i].block(0, 6+i*3, 3, 3);
        // J_linear_submix_[i] = J_temp;

        // pinocchio::WORLD
        // Eigen::MatrixXd world_J = Eigen::MatrixXd::Zero(6, pin_model_.nv);
        // pinocchio::getFrameJacobian(pin_model_, pin_data_, frame_id, pinocchio::WORLD, world_J);
        // //wb
        // world_J_linear_wb_[i] = world_J.topRows(3);
        // //leg
        // world_J_linear_leg_[i] = world_J_linear_wb_[i].block(0, 6+i*3, 3, 3);
        // //sub       
        // Eigen::MatrixXd world_J_temp = Eigen::MatrixXd::Zero(3, 12);
        // world_J_temp.block(0, 0, 3, 6) = world_J_linear_wb_[i].block(0, 0, 3, 6);
        // world_J_temp.block(0, 6, 3, 6) = world_J_linear_wb_[i].block(0, 6+3*s_idx, 3, 6);
        // world_J_linear_sub_[i] = world_J_temp;
        // //submix
        // world_J_temp = Eigen::MatrixXd::Zero(3, 12);
        // world_J_temp.block(0, 0, 3, 6) = world_J_linear_wb_[i].block(0, 0, 3, 6);
        // world_J_temp.block(0, start_col, 3, 3) = world_J_linear_wb_[i].block(0, 6+i*3, 3, 3);
        // world_J_linear_submix_[i] = world_J_temp;
    }

    // 更新子系统
    for(auto subsystems_name : subsystems_name_list_) {
        if (subsystems_name == "wb") {
            continue;
        } 
        updateSubsystem(subsystems_name, M_wb, nle_wb, inv_jac_R, x0_map);
    }
 
    return;
}

void quadrupedModel::updateSubsystem(std::string const &subsystems_name, Eigen::MatrixXd const &M_wb, 
                                     Eigen::VectorXd const &nle_wb, Eigen::MatrixXd const &inv_jac_R,
                                     std::map<std::string,std::vector<double>> const &x0_map) {
    int s_idx = 0;
    if (subsystems_name == "front") {
        s_idx = 0;
    } else if (subsystems_name == "back") {
        s_idx = 2;
    } else {
        return;
    }

    int &n_joint = config_param_.n_joint;
    int &n_contact = config_param_.n_contact;

    //适配子系统的MCG
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(6+n_joint, 6+n_joint);
    Eigen::VectorXd nle = Eigen::VectorXd::Zero(6+config_param_.n_joint);

    M.block(0, 0, 6, 6) = M_wb.block(0, 0, 6, 6);  // floating base
    M.block(6, 6, n_joint, n_joint) = M_wb.block(6+3*s_idx, 6+3*s_idx, n_joint, n_joint);
    M.block(0, 6, 6, n_joint) = M_wb.block(0, 6+3*s_idx, 6, n_joint);
    M.block(6, 0, n_joint, 6) = M_wb.block(6+3*s_idx, 0, n_joint, 6);

    nle.segment(0, 6) = nle_wb.segment(0, 6);
    nle.segment(6, 6) = nle_wb.segment(6+3*s_idx, n_joint);

    // 计算矩阵 S (12x18)
    Eigen::MatrixXd S;
    createSelectMatrix(subsystems_name, x0_map, S);

    //计算矩阵参数
    Eigen::MatrixXd inv_M = M.inverse();
    Eigen::VectorXd delta = inv_M*(-nle);

    //连续模型
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(config_param_.n_state, config_param_.n_state);
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(config_param_.n_state, config_param_.n_control);

    A.block(0, 12, 3, 3)   = Eigen::MatrixXd::Identity(3, 3);
    A.block(3, 15, 3, 3)   = inv_jac_R;
    A.block(6, 18, 6, 6)   = Eigen::MatrixXd::Identity(6, 6);
    A.block(12, 36, 12, 1) = delta;

    // case 1.1: 使用LOCAL_WORLD_ALIGNED，子系统雅可比
    A.block(24, 12, 3, 12) = J_linear_sub_[s_idx];
    A.block(27, 12, 3, 12) = J_linear_sub_[s_idx+1];

    // case 1.2: 使用LOCAL_WORLD_ALIGNED，单腿雅可比
    // A.block(24, 18, 3, 3) = J_linear_leg_[s_idx]; // 只使用每条腿自己的雅可比矩阵，腿终于能动了！！！
    // A.block(27, 21, 3, 3) = J_linear_leg_[s_idx+1];

    // case 1.3: 使用LOCAL_WORLD_ALIGNED，混合子系统雅可比
    // A.block(24, 12, 3, 12) = J_linear_submix_[s_idx];
    // A.block(27, 12, 3, 12) = J_linear_submix_[s_idx+1];

    // case 2.1: 使用WORLD，子系统雅可比
    // A.block(24, 12, 3, 12) = world_J_linear_sub_[s_idx];
    // A.block(27, 12, 3, 12) = world_J_linear_sub_[s_idx+1];

    // case 2.2: 使用WORLD，单腿雅可比
    // A.block(24, 18, 3, 3) = J_linear_wb_[s_idx].block(0, 6+s_idx*3, 3, 3);
    // A.block(27, 21, 3, 3) = J_linear_wb_[s_idx+1].block(0, 6+(s_idx+1)*3, 3, 3);

    A.block(30, 36, 6, 1)  = delta.segment(0, 6);

    Eigen::MatrixXd B_temp = inv_M*S; //12*18
    B.block(12, 0, 12, 18) = B_temp;
    B.block(30, 0, 6,  18) = B_temp.block(0, 0, 6, 18);


    // 离散化
    double const &dt = config_param_.dt; //dt==loop_dt 或者 dt>loop_dt
    // 1. 前向欧拉离散化
    // Ak_[subsystems_name] = Eigen::MatrixXd::Identity(config_param_.n_state, config_param_.n_state) + A*dt;
    // Bk_[subsystems_name] = B*dt;

    // 2. 半隐式欧拉法 semi-implicit euler
    Eigen::MatrixXd Matrix_temp = (Eigen::MatrixXd::Identity(config_param_.n_state, config_param_.n_state)-A*dt).inverse();
    Ak_[subsystems_name] = Matrix_temp;
    Bk_[subsystems_name] = Matrix_temp*B*dt;

    return;
}

std::vector<Eigen::VectorXd> quadrupedModel::updatePrediction(Eigen::VectorXd const &x0,
                                                            std::vector<Eigen::VectorXd> const &u,
                                                            std::string const &subsystems_name) {

    std::vector<Eigen::VectorXd> xtraj(config_param_.N_+1, Eigen::VectorXd::Zero(config_param_.n_state));
    if (subsystems_name == "wb") {
        return xtraj;
    }
    Eigen::VectorXd xk = x0;
    xtraj[0] = x0;

    // xtraj.push_back(x0);                                 
    for(int i=0; i<config_param_.N_; ++i) {
        xk = Ak_[subsystems_name]*xk + Bk_[subsystems_name]*u[i];
        xk(3) = normalizeAngle(xk(3));
        xk(4) = normalizeAngle(xk(4));
        xk(5) = normalizeAngle(xk(5));
        xtraj[i+1] = xk;
    }
                                        
    return xtraj;
}


void quadrupedModel::createSelectMatrix(std::string const &subsystems_name, std::map<std::string, std::vector<double>> const &x0_map,
                                        Eigen::MatrixXd &S) {
    
    int s_idx = 0;
    if (subsystems_name == "front") {
        s_idx = 0;
    } else if (subsystems_name == "back") {
        s_idx = 2;
    } else {
        return;
    }
    
    int const &n_joint = config_param_.n_joint;
    int const &n_contact_wb = config_param_.n_contact_wb;

    S = Eigen::MatrixXd::Zero(6 + n_joint, 6 + 3*n_contact_wb);
    
    // 设置 S 中与 tau 对应的部分 (后6行，前6列)
    S.block(n_joint, 0, n_joint, n_joint) = Eigen::MatrixXd::Identity(n_joint, n_joint);
    std::vector<double> contact_cmd = x0_map.at("contact_cmd");
    if (s_idx == 0) {
        for (int idx = 0; idx < 2; ++idx) { 
            Eigen::MatrixXd J_T = J_linear_sub_[idx].transpose();
            S.block(0, 6+3*idx, 12, 3) = contact_cmd[idx] * J_T;

            // Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, 12);
            // J.block(0, 0, 3, 6) = J_linear_wb_[idx].block(0, 0, 3, 6);
            // J.block(0, 6+3*idx, 3, 3) = J_linear_leg_[idx];
            // S.block(0, 6+3*idx, 12, 3) = contact_cmd[idx] * J.transpose();
        }
        for (int idx = 2; idx < 4; ++idx) { 
            Eigen::MatrixXd J_T = J_linear_sub_[idx].transpose();
            S.block(0, 6+3*idx, 6, 3) = (contact_cmd[idx] * J_T).topRows(6);

            // Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, 12);
            // J.block(0, 0, 3, 6) = J_linear_wb_[idx].block(0, 0, 3, 6);
            // S.block(0, 6+3*idx, 12, 3) = contact_cmd[idx] * J.transpose();
        }

    } else { //控制输入的顺序一直为u = [tau grf grf_aux]
        for (int idx = 0; idx < 2; ++idx) { 
            Eigen::MatrixXd J_T = J_linear_sub_[s_idx+idx].transpose();
            S.block(0, 6+3*idx, 12, 3) = contact_cmd[s_idx+idx] * J_T;

            // Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, 12);
            // J.block(0, 0, 3, 6) = J_linear_wb_[s_idx+idx].block(0, 0, 3, 6);
            // J.block(0, 6+3*idx, 3, 3) = J_linear_leg_[s_idx+idx];
            // S.block(0, 6+3*idx, 12, 3) = contact_cmd[s_idx+idx] * J.transpose();
        }
        for (int idx = 2; idx < 4; ++idx) {
            Eigen::MatrixXd J_T = J_linear_sub_[idx-s_idx].transpose();
            S.block(0, 6+3*idx, 6, 3) = (contact_cmd[idx-s_idx] * J_T).topRows(6);

            // Eigen::MatrixXd J = Eigen::MatrixXd::Zero(3, 12);
            // J.block(0, 0, 3, 6) = J_linear_wb_[idx-s_idx].block(0, 0, 3, 6);
            // S.block(0, 6+3*idx, 12, 3) = contact_cmd[idx-s_idx] * J.transpose();
        }
    }

    return;
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

#ifdef DEBUG_MODE
// 打印std::vector<double>
void debug_print(const std::vector<double>& vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i != vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

// 打印std::vector<std::vector<double>>
void debug_print(const std::vector<std::vector<double>>& mat) {
    std::cout << "[" << std::endl;
    for (size_t i = 0; i < mat.size(); ++i) {
        std::cout << "  ";
        debug_print(mat[i]);  // 调用vector<double>的print函数
    }
    std::cout << "]" << std::endl;
}

// 打印Eigen::VectorXd
void debug_print(const Eigen::VectorXd& vec) {
    std::cout << "[";
    for (int i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i != vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

// 打印Eigen::MatrixXd
void debug_print(const Eigen::MatrixXd& mat) {
    std::cout << "[" << std::endl;
    for (int i = 0; i < mat.rows(); ++i) {
        std::cout << "  [";
        for (int j = 0; j < mat.cols(); ++j) {
            std::cout << mat(i, j);
            if (j != mat.cols() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "]" << std::endl;
}
#endif