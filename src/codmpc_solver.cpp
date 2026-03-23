#include "controllers/dwmpc/codmpc_solver.hpp"

codmpcSolver::codmpcSolver()
{}

codmpcSolver::~codmpcSolver() {
#ifdef USE_FPGA
    // 销毁
    protocol_->ProtocolDestory();
#endif  
}

void codmpcSolver::init(const parameter &config_param)
{
    std::cout << "codmpcSolver initialization begins..." << std::endl;

    config_param_ = config_param;
    for(auto problem : config_param_.subsystems_name)
    {
        pdata subsystem_data{};
        data_[problem] = subsystem_data; //全部初始化为空

        std::vector<Eigen::VectorXd> u0(config_param_.N_, Eigen::VectorXd::Zero(config_param_.n_control));
        u_[problem] = u0;
        u_ref_[problem] = u0;

        std::vector<Eigen::VectorXd> x0(config_param_.N_+1, Eigen::VectorXd::Zero(config_param_.n_state));
        x_[problem] = x0;
        x_ref_[problem] = x0;

        x0_[problem] = Eigen::VectorXd::Zero(config_param_.n_state);

    }
    
    Q_ = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(Eigen::VectorXd::Zero(config_param_.n_state));
    R_ = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(Eigen::VectorXd::Zero(config_param_.n_control));

    quadruped_model_.modelInit(config_param);

#ifdef USE_FPGA
    protocolInit();
#endif

// #ifdef DEBUG_MODE
//     data_logger_.init("quadruped_data.csv");
// #endif

    std::cout << "codmpcSolver initialized!!!" << std::endl;

    return;
}

void codmpcSolver::solve( bool &do_init,
                    const std::map<std::string,std::vector<double>> &x0_map,
                    const std::map<std::string,std::vector<std::vector<double>>> &ref,
                    const std::map<std::string,std::vector<std::vector<double>>> &param,
                    const std::map<std::string,std::vector<double>> &weight_vec) 
{   
    // init data to the reference 
    if (do_init)
    {    
        for(auto problem : config_param_.subsystems_name) //对所有subsystem初始化
        {   
            // init to refernce
            data_[problem].p = ref.at("p");
            data_[problem].quat = ref.at("quat");
            data_[problem].rpy = ref.at("rpy");
            data_[problem].dp = ref.at("dp");
            data_[problem].omega = ref.at("omega");
            for (int k{0};k<config_param_.N_+1;k++)
            {   
                std::vector<double> q,dq,tau,grf,foot;
                for (auto idx : config_param_.subsystems_map_joint[problem])
                {
                    q.push_back(ref.at("q")[k][idx]);
                    dq.push_back(ref.at("dq")[k][idx]);
                    tau.push_back(ref.at("tau")[k][idx]);
                }
                for (auto idx : config_param_.subsystems_map_contact[problem])
                {
                    grf.push_back(ref.at("grf")[k][3*idx]);
                    grf.push_back(ref.at("grf")[k][3*idx+1]);
                    grf.push_back(ref.at("grf")[k][3*idx+2]);

                    foot.push_back(ref.at("foot")[k][3*idx]);
                    foot.push_back(ref.at("foot")[k][3*idx+1]);
                    foot.push_back(ref.at("foot")[k][3*idx+2]);
                }
                data_[problem].tau.push_back(tau);
                data_[problem].grf.push_back(grf);
                data_[problem].foot.push_back(foot);
                data_[problem].q.push_back(q);
                data_[problem].dq.push_back(dq);
                data_[problem].dual.push_back(std::vector<double>(6,0));
                data_[problem].residual.push_back(std::vector<double>(6,0));
            }
        }
    }

    // main loop  (number of iteration)
    //problem loop
    // ============       MODEL       ============
    tm_.start("modelUpdate");
    quadruped_model_.modelUpdate(x0_map); // 放在循环外面
    tm_.stop("modelUpdate");
    tm_.print("modelUpdate");

    for (auto problem : config_param_.subsystems_name)
    {

        int counter = 0;

        // ============ INITAIAL CONDITION ============ 
        x0_[problem](0) = x0_map.at("p")[0];
        x0_[problem](1) = x0_map.at("p")[1];
        x0_[problem](2) = x0_map.at("p")[2];

        x0_[problem](3) = normalizeAngle(x0_map.at("rpy")[2]);
        x0_[problem](4) = normalizeAngle(x0_map.at("rpy")[1]);
        x0_[problem](5) = normalizeAngle(x0_map.at("rpy")[0]);
        // problem_initial_condition.push_back(normalizeAngle(x0_map.at("rpy")[0] - ref.at("rpy")[0][0])); // 姿态欧拉角ref设置为0，由于欧拉角有过圈问题，在这里先算好误差
        // problem_initial_condition.push_back(normalizeAngle(x0_map.at("rpy")[1] - ref.at("rpy")[0][1]));
        // problem_initial_condition.push_back(normalizeAngle(x0_map.at("rpy")[2] - ref.at("rpy")[0][2]));

        counter = 0;
        for(auto idx : config_param_.subsystems_map_joint[problem]) //循环12次
        {
            x0_[problem](6+counter) = x0_map.at("q")[idx];
            ++counter;
        }

        x0_[problem](18) = x0_map.at("dp")[0];
        x0_[problem](19) = x0_map.at("dp")[1];
        x0_[problem](20) = x0_map.at("dp")[2];

        x0_[problem](21) = x0_map.at("omega")[2];
        x0_[problem](22) = x0_map.at("omega")[1];
        x0_[problem](23) = x0_map.at("omega")[0];

        counter=0;
        for(auto idx : config_param_.subsystems_map_joint[problem]) //循环12次
        {
            x0_[problem](24+counter) = x0_map.at("dq")[idx];
            ++counter;
        }
        
        counter=0;
        for (auto idx : config_param_.subsystems_map_contact[problem]) //循环12次
        {
            x0_[problem](36+counter) = x0_map.at("foot")[3*idx];
            x0_[problem](37+counter) = x0_map.at("foot")[3*idx+1];
            x0_[problem](38+counter) = x0_map.at("foot")[3*idx+2];
            counter+=3;
        }

        x0_[problem](48) = 1.0;

        //std::cout << std::endl;

        ////  ============ REFERENCE  ============
        // horizon loop
        for (auto k{0};k<config_param_.N_+1; k++)
        {   
            //set p,quat 
            x_ref_[problem][k](0) = ref.at("p")[k][0];
            x_ref_[problem][k](1) = ref.at("p")[k][1];
            x_ref_[problem][k](2) = ref.at("p")[k][2];

            x_ref_[problem][k](3) = ref.at("rpy")[k][2];
            x_ref_[problem][k](4) = ref.at("rpy")[k][1];
            x_ref_[problem][k](5) = ref.at("rpy")[k][0];
            // ref_k.push_back(0); // 姿态欧拉角ref设置为0，由于欧拉角有过圈问题，在这里先算好误差
            // ref_k.push_back(0);
            // ref_k.push_back(0);

            // set q
            counter=0;
            for(auto idx : config_param_.subsystems_map_joint[problem]) //循环12次
            {
                x_ref_[problem][k](6+counter) = ref.at("q")[k][idx];
                ++counter;
            }

            // set dp omega
            x_ref_[problem][k](18) = ref.at("dp")[k][0];
            x_ref_[problem][k](19) = ref.at("dp")[k][1];
            x_ref_[problem][k](20) = ref.at("dp")[k][2];

            x_ref_[problem][k](21) = ref.at("omega")[k][2];
            x_ref_[problem][k](22) = ref.at("omega")[k][1];
            x_ref_[problem][k](23) = ref.at("omega")[k][0];

            // set dq
            counter=0;
            for(auto idx : config_param_.subsystems_map_joint[problem]) //循环12次
            {
                x_ref_[problem][k](24+counter) = ref.at("dq")[k][idx];
                ++counter;
            }
            // set foot
            counter=0;
            for (auto idx : config_param_.subsystems_map_contact[problem]) //循环12次
            {
                x_ref_[problem][k](36+counter) = ref.at("foot")[k][3*idx];
                x_ref_[problem][k](37+counter) = ref.at("foot")[k][3*idx+1];
                x_ref_[problem][k](38+counter) = ref.at("foot")[k][3*idx+2];
                counter+=3;
            }

            x_ref_[problem][k](48) = 1.0;

            ////  ============ REFERENCE  U ============
            if (k < config_param_.N_) { // u N维
                // set tau
                counter=0;
                for(auto idx : config_param_.subsystems_map_joint[problem]) // 循环12次
                {
                    u_ref_[problem][k](counter) = ref.at("tau")[k][idx];
                    ++counter;
                }

                // set grf //这里和原代码不同，我们只优化grf而不是grf_wb，因此只给当前子系统赋值即可
                counter=0;
                for(auto idx : config_param_.subsystems_map_contact[problem]) // 循环12次
                {
                    u_ref_[problem][k](12+counter) = ref.at("grf")[k][3*idx];
                    u_ref_[problem][k](13+counter) = ref.at("grf")[k][3*idx+1];
                    u_ref_[problem][k](14+counter) = ref.at("grf")[k][3*idx+2];
                    counter+=3;              
                }
            }

            ////  ============ WEIGHT  ============                
            if(do_init) //本来是每个预测step都有一个权重，这里就不改了
            {
                // weight p 
                Q_.diagonal()[0] = weight_vec.at("p")[0];
                Q_.diagonal()[1] = weight_vec.at("p")[1];
                Q_.diagonal()[2] = weight_vec.at("p")[2];

                // weight quat
                Q_.diagonal()[3] = weight_vec.at("quat")[2];
                Q_.diagonal()[4] = weight_vec.at("quat")[1];
                Q_.diagonal()[5] = weight_vec.at("quat")[0];

                // weight q
                counter = 0;
                for(auto idx : config_param_.subsystems_map_joint[problem]) // 实际循环12次
                {
                    Q_.diagonal()[6+counter] = weight_vec.at("q")[0];
                    counter++;
                }
            
                // weight dp
                Q_.diagonal()[18] = weight_vec.at("dp")[0];
                Q_.diagonal()[19] = weight_vec.at("dp")[1];
                Q_.diagonal()[20] = weight_vec.at("dp")[2];

                // weight omega
                Q_.diagonal()[21] = weight_vec.at("omega")[2];
                Q_.diagonal()[22] = weight_vec.at("omega")[1];
                Q_.diagonal()[23] = weight_vec.at("omega")[0];

                // weight dq
                counter = 0;
                for(auto idx : config_param_.subsystems_map_joint[problem]) // 实际循环12次
                {
                    Q_.diagonal()[24+counter] = weight_vec.at("dq")[0];
                    counter++;
                }

                // weight foot
                counter = 0;
                for(auto idx : config_param_.subsystems_map_contact[problem]) // 实际循环4*3次
                {   
                    if (param.at("contact_seq")[k][idx] == 1)
                    {
                        Q_.diagonal()[36 + counter] = weight_vec.at("foot_stance")[0];
                        Q_.diagonal()[37 + counter] = weight_vec.at("foot_stance")[1];
                        Q_.diagonal()[38 + counter] = weight_vec.at("foot_stance")[2];
                    }
                    else
                    {
                        Q_.diagonal()[36 + counter] = weight_vec.at("foot_swing")[0]; //分配摆动腿或站立腿权重
                        Q_.diagonal()[37 + counter] = weight_vec.at("foot_swing")[1];
                        Q_.diagonal()[38 + counter] = weight_vec.at("foot_swing")[2];
                    }
                    counter+=3;
                }

                // weight constant 1
                Q_.diagonal()[48] = 0;

                // weight tau
                counter = 0;
                for(auto idx : config_param_.subsystems_map_joint[problem]) // 实际循环12次
                {
                    R_.diagonal()[counter] = weight_vec.at("tau")[0];
                    counter++;
                }

                // weight grf grf_aux
                counter = 0;
                for(auto idx : config_param_.subsystems_map_contact["wb"])
                {
                    R_.diagonal()[12+counter] = weight_vec.at("grf")[0];
                    R_.diagonal()[13+counter] = weight_vec.at("grf")[0];                            
                    R_.diagonal()[14+counter] = weight_vec.at("grf")[0];
                    counter+=3;
                }

                // gamma
                gamma_ = weight_vec.at("gamma")[0];
            }              
        }
        // pass to the codmpc sovler

#ifdef USE_FPGA
        dataSend(x0_map, problem);
#endif

#ifdef USE_QPOASES 
        bool success = qpOASESsolve(x0, x0_map, x_ref_, u_ref, problem);
        if (!success) {
            std::cout << "MPC求解失败！" << std::endl;
        }
#endif  


#ifdef USE_HPIPM
        bool success = hpipmSolve(x0_map, problem);
        if (!success) {
            std::cout << "MPC求解失败！" << std::endl;
        }
#endif  

    }

#ifdef USE_FPGA
    if (!is_front_solved) {
        std::cout << "子问题front未求解成功！！！" << std::endl;
    }
    if (!is_back_solved) {
        std::cout << "子问题back未求解成功！！！" << std::endl;
    }
#endif

    for (auto problem : config_param_.subsystems_name)
    {   
        // update state from solution

#ifdef USE_HPIPM
        std::vector<Eigen::VectorXd> &x = x_[problem];
#else
        std::vector<Eigen::VectorXd> x = quadruped_model_.updatePrediction(x0_[problem], u_[problem], problem);
#endif
        
        int n_joints {static_cast<int>(config_param_.subsystems_map_joint[problem].size())}; //6
        int counter = 0;
        //update data state
        for (int k{0};k<config_param_.N_+1;k++)
        {   
            //p 
            data_[problem].p[k][0] = x[k](0);
            data_[problem].p[k][1] = x[k](1);
            data_[problem].p[k][2] = x[k](2);

            //rpy quat
            data_[problem].rpy[k][0] = x[k](5); //data永远是rpy x3~5: yaw pitch roll
            data_[problem].rpy[k][1] = x[k](4);
            data_[problem].rpy[k][2] = x[k](3);

            Eigen::Quaterniond quat = rpyToquat(Eigen::Vector3d(x[k](5), x[k](4), x[k](3)));

            data_[problem].quat[k][0] = quat.x();
            data_[problem].quat[k][1] = quat.y();
            data_[problem].quat[k][2] = quat.z();
            data_[problem].quat[k][3] = quat.w();

            //q
            counter = 0;
            for(auto idx : config_param_.subsystems_map_joint[problem]) //12个循环
            {
                data_["wb"].q[k][idx] = x[k](6+counter);  // data_["wb"]放的是当前及预测状态，但是没放全   
                counter++;
            }
            
            //dp
            data_[problem].dp[k][0] = x[k](18);
            data_[problem].dp[k][1] = x[k](19);
            data_[problem].dp[k][2] = x[k](20);

            //omega
            data_[problem].omega[k][0] = x[k](23);
            data_[problem].omega[k][1] = x[k](22);
            data_[problem].omega[k][2] = x[k](21);

            //dq
            counter = 0;
            for(auto idx : config_param_.subsystems_map_joint[problem])  //6个循环
            {
                data_["wb"].dq[k][idx] = x[k](24+counter);
                counter++;
            }

            //foot
            counter = 0;
            for (auto idx : config_param_.subsystems_map_contact[problem]) //循环2*3次
            {
                data_["wb"].foot[k][3*idx]   = x[k](36+counter);
                data_["wb"].foot[k][3*idx+1] = x[k](37+counter);
                data_["wb"].foot[k][3*idx+2] = x[k](38+counter);
                counter += 3;
            }

            // consensus: dp omega 不记录，在下面更新，放在data_["wb"]中而不是data_[problem]中

            //control input
            if (k < config_param_.N_) {
                //tau
                counter = 0;
                for(auto idx : config_param_.subsystems_map_joint[problem])
                {
                    data_["wb"].tau[k][idx] = u_[problem][k](counter);
                    counter++;
                }

                //grf
                counter = 0;
                for(auto idx : config_param_.subsystems_map_contact[problem])
                {
                    data_["wb"].grf[k][3*idx] = u_[problem][k](n_joints+3*counter);
                    data_["wb"].grf[k][3*idx+1] = u_[problem][k](n_joints+3*counter+1);
                    data_["wb"].grf[k][3*idx+2] = u_[problem][k](n_joints+3*counter+2);
                    counter++;
                }
            }
        }
    }
    // check stopping criteria

    do_init = false;
}

void codmpcSolver::prepare()
{
    // TODO
    return;
}
void codmpcSolver::getControl(std::vector<double> &des_q,std::vector<double> &des_dq,std::vector<double> &des_tau)
{   
    des_q = data_["wb"].q[1];
    des_dq = data_["wb"].dq[1];
    des_tau = data_["wb"].tau[0];

}
void codmpcSolver::getData(std::map<std::string,pdata> &data)
{  
    data = data_;
}

#ifdef USE_FPGA
// 将字节数组转换为 std::string，发送数据时使用
std::string codmpcSolver::ByteArrayToString(const std::vector<uint8_t>& byteArray) {
    return std::string(reinterpret_cast<const char*>(byteArray.data()), byteArray.size());
}

// 将 std::string 转换为字节数组，接收数据时使用
std::vector<uint8_t> codmpcSolver::StringToByteArray(const std::string& str) {
    std::vector<uint8_t> byteArray(str.begin(), str.end());
    return byteArray;
}

void codmpcSolver::dataRecvCallback(const std::string& data) {

    std::vector<uint8_t> recv_buffer = StringToByteArray(data);

    // 长度校验
    int const total_byte_length = 2564;// (20*10+40*11)*4+4=2564
    if (recv_buffer.size() != total_byte_length) {
        std::cout << "接收数据长度错误！！！期望 "<< total_byte_length << " byte, 收到 " <<recv_buffer.size()<<" byte！！！"<< std::endl;
    }

    // 定义帧头帧尾
    std::array<uint8_t, 2> const FRAME_HEADER = {0xAA, 0xBB};
    std::array<uint8_t, 2> const FRAME_FOOTER = {0xCC, 0xDD};

    // 帧头校验
    if (*recv_buffer.begin() != FRAME_HEADER[0] || *(recv_buffer.begin()+1) != FRAME_HEADER[1]) {
        std::cout << "接收数据帧头错误！！！" << std::endl;
        return;
    }

    // 帧尾校验
    if (*(recv_buffer.end()-2) != FRAME_FOOTER[0] || *(recv_buffer.end()-1) != FRAME_FOOTER[1]) {
        std::cout << "接收数据帧尾错误！！！" << std::endl;
        return;
    }

    // 检查字节数是否为4的倍数（每个float占4字节）
    if ((recv_buffer.size()-4) % 4 != 0) {
        std::cerr << "警告：接收的字节数不是4的倍数，可能存在数据不完整！" << std::endl;
    }

    // 计算可转换的float数量
    size_t float_count = (recv_buffer.size()-4) / 4;
    Eigen::VectorXf result(float_count);

    // 遍历字节流，每4字节转换为一个float（跳过帧头帧尾）
    for (size_t i = 2; i < float_count; ++i) {
        // 获取当前组的起始地址（第i个float的第1个字节）
        uint8_t* byte_ptr = &recv_buffer[i * 4];
        // 将uint8_t*转换为float*，解引用得到float值
        float* float_ptr = reinterpret_cast<float*>(byte_ptr);
        result << *float_ptr;
    }

    // 接收数据
    int const &N = config_param_.N_;
    int const &nx = config_param_.n_state;
    int const &nu = config_param_.n_control;
    int const nx_send = 40;
    int const nu_send = 20;
    int const start_idx_x = nu_send*N;

    std::string subsystems_name;
    if(!is_front_solved) {
        subsystems_name = "front";
        is_front_solved = true;
    } else if (!is_back_solved) {
        subsystems_name = "back";
        is_back_solved = true;
    } else {
        std::cout << "错误！！！接收数据时is_front_solved和is_back_solved全部是1！！！" << std::endl;
        return;
    }

    for (int i = 0; i < N; ++i) {
        u_[subsystems_name][i] = result.segment(i*nu_send, nu).cast<double>();
        x_[subsystems_name][i] = result.segment(start_idx_x+i*nx_send, nx).cast<double>();
    }
    x_[subsystems_name][N] = result.segment(start_idx_x+N*nx_send, nx).cast<double>(); //状态多一维

    return;
}

// 模板辅助函数：将Eigen矩阵/向量的float数据转换为字节并添加到buffer
template <typename T>
void codmpcSolver::appendEigenData(const T& data, std::vector<uint8_t>& buffer) {
    // 确保数据非空
    if (data.size() == 0) return;
    
    // 获取数据起始地址（float*）
    const float* float_ptr = data.data();
    // 计算总字节数（每个float占4字节）
    size_t total_bytes = data.size() * 4;
    // 转换为uint8_t指针以按字节访问
    const uint8_t* byte_ptr = reinterpret_cast<const uint8_t*>(float_ptr);
    
    // 将所有字节添加到buffer末尾
    buffer.insert(buffer.end(), byte_ptr, byte_ptr + total_bytes);
}

void codmpcSolver::protocolInit() {
    proto_config_.protocol_type_ = fish_protocol::PROTOCOL_TYPE::SERIAL;
    proto_config_.serial_baut_ = 115200;
    proto_config_.serial_address_ = "/dev/ttyUSB0";

    // 初始化
    protocol_ = GetProtocolByConfig(proto_config_);

    // 设置接收数据回调函数
    // 方案1：用std::bind（需包含 <functional> 头文件）
    // protocol_->SetDataRecvCallback(std::bind(&codmpcSolver::dataRecvCallback, this, std::placeholders::_1));

    // 方案2：用lambda（更简洁）
    protocol_->SetDataRecvCallback([this](const std::string& data) {
        this->dataRecvCallback(data);
    });

    return;
}

bool codmpcSolver::dataSend(std::map<std::string,std::vector<double>> const &x0_map,
                            std::string const &subsystems_name) {      
    // setup QP
    int s_idx = 0;
    if (subsystems_name == "front") {
        s_idx = 0;
        is_front_solved = false;
    } else if (subsystems_name == "back") {
        s_idx = 2;
        is_back_solved = false;
    } else {
        return false;
    }

    int const &N = config_param_.N_;
    int const &nx = config_param_.n_state;
    int const &nu = config_param_.n_control;
    int const nx_send = 40;
    int const nu_send = 20;
    int const nc_send = 8;

    // dynamics
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A = Eigen::MatrixXf::Zero(nx_send, nx_send);
    A.block(0, 0, nx, nx) = quadruped_model_.Ak_[subsystems_name].cast<float>();
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> B = Eigen::MatrixXf::Zero(nx_send, nu_send);
    B.block(0, 0, nx, nu) = quadruped_model_.Bk_[subsystems_name].cast<float>();

    // constraints
    double const epsilon = 5e-3;
    int n_noslip_constrain = 2*3;
    std::vector<double> const contact_cmd = x0_map.at("contact_cmd");
    Eigen::MatrixXd J_matrix = Eigen::MatrixXd::Zero(n_noslip_constrain, 12);
    J_matrix.block(0, 0, 3, 12) = contact_cmd[s_idx]*quadruped_model_.J_linear_sub_[s_idx];
    J_matrix.block(3, 0, 3, 12) = contact_cmd[s_idx+1]*quadruped_model_.J_linear_sub_[s_idx+1];
    Eigen::MatrixXd J_select = Eigen::MatrixXd::Zero(n_noslip_constrain, nx);
    J_select.block(0, 12, n_noslip_constrain, 12) = J_matrix;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> C = Eigen::MatrixXf::Zero(nc_send, nx_send);
    C.block(0, 0, n_noslip_constrain, nx) = J_select.cast<float>();

    // x0
    Eigen::VectorXf x0 = x0_[subsystems_name].cast<float>();

    // x_ref
    Eigen::VectorXf x_ref = Eigen::VectorXf::Zero(nx_send*(N+1));
    for (int i=0; i<=N; ++i) {
        x_ref.segment(i*nx_send, nx) = x_ref_[subsystems_name][i].cast<float>();
    }

    // u_ref
    Eigen::VectorXf u_ref = Eigen::VectorXf::Zero(nu_send*N);
    for (int i=0; i<N; ++i) {
        x_ref.segment(i*nu_send, nu) = u_ref_[subsystems_name][i].cast<float>();
    }    

    // B的伪逆
    Eigen::MatrixXf B_pinv = B.completeOrthogonalDecomposition().pseudoInverse().cast<float>();

    // 构建发送缓冲区
    // 定义帧头帧尾
    std::array<uint8_t, 2> const FRAME_HEADER = {0xAA, 0xBB};
    std::array<uint8_t, 2> const FRAME_FOOTER = {0xCC, 0xDD};

    std::vector<uint8_t> send_buffer;
    // 插入帧头（在数据最前面）
    send_buffer.insert(send_buffer.begin(), FRAME_HEADER.begin(), FRAME_HEADER.end());
    // 插入数据
    appendEigenData(A, send_buffer);
    appendEigenData(B, send_buffer);
    appendEigenData(C, send_buffer);
    appendEigenData(x0, send_buffer);
    appendEigenData(x_ref, send_buffer);
    appendEigenData(u_ref, send_buffer);
    appendEigenData(B_pinv, send_buffer);
    // 插入帧尾（在数据最后面）
    send_buffer.insert(send_buffer.end(), FRAME_FOOTER.begin(), FRAME_FOOTER.end());

    // 发送数据
    protocol_->ProtocolSendRawData(ByteArrayToString(send_buffer));

    return true;
}

#endif

#ifdef USE_HPIPM

bool codmpcSolver::hpipmSolve(std::map<std::string,std::vector<double>> const &x0_map,
                              std::string const &subsystems_name) {
    // setup QP
    int const &N = config_param_.N_;
    int const &nx = config_param_.n_state;
    int const &nu = config_param_.n_control;
    int const &n_contact_wb = config_param_.n_contact_wb;
    Eigen::VectorXd const &x0 = x0_[subsystems_name];
    std::vector<Eigen::VectorXd> const &x_ref = x_ref_[subsystems_name];
    std::vector<Eigen::VectorXd> const &u_ref = u_ref_[subsystems_name];

    std::vector<hpipm::OcpQp> qp(N+1);

    // dynamics
    Eigen::MatrixXd &A = quadruped_model_.Ak_[subsystems_name];
    Eigen::MatrixXd &B = quadruped_model_.Bk_[subsystems_name];
    const Eigen::VectorXd b = Eigen::VectorXd::Zero(nx);
    for (int i=0; i<N; ++i) { //0～N-1
        qp[i].A = A;
        qp[i].B = B;
        qp[i].b = b;
    }

    // cost
    Eigen::MatrixXd Q(nx, nx), S(nu, nx), R(nu, nu);
    Q.setZero(); Q.diagonal() = Q_.diagonal();
    S.setZero();
    R.setZero(); R.diagonal() << R_.diagonal();
    // const Eigen::VectorXd q = - Q * x_ref;
    // const Eigen::VectorXd r = Eigen::VectorXd::Zero(nu);
    Eigen::VectorXd q = Eigen::VectorXd::Zero(nx);
    Eigen::VectorXd r = Eigen::VectorXd::Zero(nu);
    for (int i=0; i<N; ++i) { //0～N-1
        q = - Q * x_ref[i];
        r = - R * u_ref[i];
        qp[i].Q = Q;
        qp[i].R = R;
        qp[i].S = S;
        qp[i].q = q;
        qp[i].r = r;
        Q *= gamma_;
        // R *= gamma_;
    }
    q = - Q * x_ref[N];
    qp[N].Q = Q;
    qp[N].q = q;

    // constraints
    constrains_ = 0;

    /////////////////// constrain 1: foot noslip
    double const epsilon = 5e-3;
    int n_noslip_constrain = 4*3;
    constrains_ += n_noslip_constrain;

    std::vector<double> const contact_cmd = x0_map.at("contact_cmd");
    Eigen::MatrixXd J_matrix = Eigen::MatrixXd::Zero(n_noslip_constrain, 18);
    J_matrix.block(0, 0, 3, 18) = contact_cmd[0]*quadruped_model_.J_linear_wb_[0];
    J_matrix.block(3, 0, 3, 18) = contact_cmd[1]*quadruped_model_.J_linear_wb_[1];
    J_matrix.block(6, 0, 3, 18) = contact_cmd[2]*quadruped_model_.J_linear_wb_[2];
    J_matrix.block(9, 0, 3, 18) = contact_cmd[3]*quadruped_model_.J_linear_wb_[3];

    Eigen::MatrixXd J_select = Eigen::MatrixXd::Zero(n_noslip_constrain, nx);
    J_select.block(0, 18, n_noslip_constrain, 18) = J_matrix;
    Eigen::VectorXd vec_foot_vel_max = epsilon*Eigen::VectorXd::Ones(n_noslip_constrain);
    Eigen::VectorXd vec_foot_vel_min = -vec_foot_vel_max;

    //////////////////// constrain 2: friction cone
    double const mu = 0.5;
    double const fz_max = 500;
    double const fz_min = 0;

    int n_friction_cone_constrain = 4*5;
    constrains_ += n_friction_cone_constrain;
    Eigen::MatrixXd friction_matrix_block(5, 3);
    friction_matrix_block << 1,  0, mu,
                            -1,  0, mu,
                             0,  1, mu,
                             0, -1, mu,
                             0,  0, 1;

    Eigen::MatrixXd friction_matrix = Eigen::MatrixXd::Zero(n_friction_cone_constrain, nu);
    for (int i=0; i<n_contact_wb; ++i) {
        friction_matrix.block(5*i, 12+3*i, 5, 3) = friction_matrix_block;
    }

    Eigen::VectorXd vec_friction_min(n_friction_cone_constrain);
    Eigen::VectorXd vec_friction_max(n_friction_cone_constrain);
    vec_friction_min << 0, 0, 0, 0, fz_min,
                        0, 0, 0, 0, fz_min,
                        0, 0, 0, 0, fz_min,
                        0, 0, 0, 0, fz_min;
    vec_friction_max << fz_max, fz_max, fz_max, fz_max, fz_max,
                        fz_max, fz_max, fz_max, fz_max, fz_max,
                        fz_max, fz_max, fz_max, fz_max, fz_max,
                        fz_max, fz_max, fz_max, fz_max, fz_max;

#if 1
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(constrains_, nx);
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(constrains_, nu);
    Eigen::VectorXd lg = Eigen::VectorXd::Zero(constrains_);
    Eigen::VectorXd ug = Eigen::VectorXd::Zero(constrains_);

    C.topRows(n_noslip_constrain) = J_select;          // 前 n_c 行对应 Cx 的约束
    C.bottomRows(n_friction_cone_constrain).setZero(); // 后 n_d 行不涉及 x（对应 Du 的约束）
    D.topRows(n_noslip_constrain).setZero();           // 前 n_c 行不涉及 u（对应 Cx 的约束）
    D.bottomRows(n_friction_cone_constrain) = friction_matrix;    // 后 n_d 行对应
    lg.head(n_noslip_constrain) = vec_foot_vel_min; // 前 n_c 个元素为 lgc
    lg.tail(n_friction_cone_constrain) = vec_friction_min; // 后 n_d 个元素为 lgd
    ug.head(n_noslip_constrain) = vec_foot_vel_max; // 前 n_c 个元素为 ugc
    ug.tail(n_friction_cone_constrain) = vec_friction_max; // 后 n_d 个元素为 ugd

    qp[0].C = Eigen::MatrixXd::Zero(n_friction_cone_constrain, nx);
    qp[0].D = friction_matrix;  
    qp[0].lg = vec_friction_min;
    qp[0].ug = vec_friction_max;
    for (int i = 1; i < N; ++i) { // 注意：状态1～N，控制输入0~N-1
        // 设置合并后的约束矩阵
        qp[i].C = C;  // C_total x + D_total u 的状态部分矩阵
        qp[i].D = D;  // C_total x + D_total u 的输入部分矩阵

        // 设置合并后的上下界
        qp[i].lg = lg;      // 总下界：lg <= C_total x + D_total u
        qp[i].ug = ug;      // 总上界：C_total x + D_total u <= ug
    }
    qp[N].C = J_select;
    qp[N].D = Eigen::MatrixXd::Zero(n_noslip_constrain, nu);  
    qp[N].lg = vec_foot_vel_min;
    qp[N].ug = vec_foot_vel_max;
#else //只加摩擦锥约束

    for (int i = 0; i < N; ++i) { // 注意：状态1～N，控制输入0~N-1
        // 设置合并后的约束矩阵
        qp[i].C = Eigen::MatrixXd::Zero(n_friction_cone_constrain, nx);
        qp[i].D = friction_matrix;  // C_total x + D_total u 的输入部分矩阵

        // 设置合并后的上下界
        qp[i].lg = vec_friction_min;      // 总下界：lg <= C_total x + D_total u
        qp[i].ug = vec_friction_max;      // 总上界：C_total x + D_total u <= ug
    }

#endif

    hpipm::OcpQpIpmSolverSettings solver_settings;
    solver_settings.mode = hpipm::HpipmMode::SpeedAbs;
    solver_settings.iter_max = 100;
    solver_settings.alpha_min = 1e-8;
    solver_settings.mu0 = 1e2;
    solver_settings.tol_stat = 1e-04;
    solver_settings.tol_eq = 1e-04;
    solver_settings.tol_ineq = 1e-04;
    solver_settings.tol_comp = 1e-04;
    solver_settings.reg_prim = 1e-12;
    solver_settings.warm_start = 0;
    solver_settings.pred_corr = 1;
    solver_settings.ric_alg = 1;
    solver_settings.split_step = 1;

    std::vector<hpipm::OcpQpSolution> solution(N+1);
    hpipm::OcpQpIpmSolver solver(qp, solver_settings);

    for (int i=0; i<N; ++i) { //热启动部分，TODO
        solution[i].x = x_[subsystems_name][i];
        solution[i].u = u_[subsystems_name][i];
    }
    solution[N].x = x_[subsystems_name][N];

    tm_.start("hpipm");
    auto status = solver.solve(x0, qp, solution); //求解MPC问题
    tm_.stop("hpipm");

    tm_.print("hpipm");

    if (status == hpipm::HpipmStatus::Success) {
        // 保存控制和状态序列
        for (int i = 0; i < N; ++i) {
            u_[subsystems_name][i] = solution[i].u;
            x_[subsystems_name][i] = solution[i].x;
        }
        x_[subsystems_name][N] = solution[N].x; //状态多一维

        return true;
    } else {
        std::cerr << "HPIPM solving failed! Error code: " << status << std::endl;
    }

    return false;
}
#endif

#ifdef USE_QPOASES

// 构建总权重矩阵 (Q_total和R_total)，预先计算，只算一次
void codmpcSolver::buildTotalWeightMatrices() {

    int const &N = config_param_.N_;
    int const &n = config_param_.n_state;
    int const &m = config_param_.n_control;

    // Q_total = diag(Q, Q, ..., Q, P)（前N-1个Q，最后1个P）
    Eigen::VectorXd q_total_diag(n*N);
    for (int k = 0; k < N-1; ++k) {
        q_total_diag.segment(k*n, n) = Q_.diagonal();
    }
    q_total_diag.segment((N-1)*n, n) = Q_.diagonal();
    Q_total_.diagonal() = q_total_diag;
    
    // R_total = diag(R, R, ..., R)（共N个R）
    Eigen::VectorXd r_total_diag(m*N);
    for (int k = 0; k < N; ++k) {
        r_total_diag.segment(k*m, m) = R_.diagonal();
    }
    R_total_.diagonal() = r_total_diag;

    R_total_dense_ = R_total_.toDenseMatrix();

    return;
}

// 构造F矩阵: [I; A; A^2; ...; A^N]
void codmpcSolver::buildFMatrix(Eigen::MatrixXd &F, Eigen::MatrixXd const &A) {

    int const &N = config_param_.N_;
    int const &n = config_param_.n_state;

    F = Eigen::MatrixXd::Zero(n*N, n);

    Eigen::MatrixXd A_pow = A;  // 从A^1开始（对应x1 = A^1 x0 + ...）
    
    for (int k = 0; k < N; ++k) {  // k=0对应x1, ..., k=N-1对应xN
        F.block(k*n, 0, n, n) = A_pow;
        A_pow = A * A_pow;  // 计算A^(k+2)
    }
    
    return;
}
    
// 构造Phi矩阵
void codmpcSolver::buildPhiMatrix(Eigen::MatrixXd &Phi, Eigen::MatrixXd const &A, Eigen::MatrixXd const &B) {

    int const &N = config_param_.N_;
    int const &n = config_param_.n_state;
    int const &m = config_param_.n_control;

    Phi = Eigen::MatrixXd::Zero(n*N, m*N);
    
    for (int k = 0; k < N; ++k) {  // 行块：对应< N; ++k) {  // 行块：对应x_{k+1}（k=0→x1, ..., k=N-1→xN）
        Eigen::MatrixXd A_pow = Eigen::MatrixXd::Identity(n, n);  // 初始为A^0
        // 列块：i从k递减到0，确保A_pow = A^(k-i)
        for (int i = k; i >= 0; --i) {  
            int row_start = k * n;       // 当前行块起始索引
            int col_start = i * m;       // 当前列块起始索引（对应u_i）
            Phi.block(row_start, col_start, n, m) = A_pow * B;
            A_pow = A * A_pow;  // A_pow从A^0 → A^1 → ... → A^k
        }
    }
    
    return;
}

void codmpcSolver::qpOASESinit() {

    //构造大权重矩阵
    buildTotalWeightMatrices();

    return;
}

// 计算MPC问题转化为QP问题时的H矩阵和g向量
// 使用Eigen::DiagonalMatrix存储Q和R，利用Eigen内部优化
void codmpcSolver::computeQPmatrices(std::string const &subsystems_name,
    Eigen::VectorXd const &x0, std::map<std::string,std::vector<double>> const &x0_map,
    std::vector<Eigen::VectorXd> const &x_ref,
    std::vector<Eigen::VectorXd> const &u_ref,
    Eigen::MatrixXd& H, Eigen::VectorXd& g, 
    Eigen::MatrixXd& Ac, Eigen::VectorXd& lbAc, Eigen::VectorXd& ubAc) {

    // 参数设置
    int s_idx = 0;
    if (subsystems_name == "front") {
        s_idx = 0;
    } else if (subsystems_name == "back") {
        s_idx = 2;
    } else {
        return;
    }
    int const &N = config_param_.N_;
    int const &n = config_param_.n_state;
    int const &m = config_param_.n_control;
    int const &n_contact_wb = config_param_.n_contact_wb;
    int const total_n = n * N;
    int const total_m = m * N;

    Eigen::MatrixXd const &A = quadruped_model_.Ak_[subsystems_name];
    Eigen::MatrixXd const &B = quadruped_model_.Bk_[subsystems_name];

    // 构建矩阵
    Eigen::MatrixXd F;
    Eigen::MatrixXd Phi;    
    buildFMatrix(F, A);
    buildPhiMatrix(Phi, A, B);
        
    // 构建参考向量
    Eigen::VectorXd X_ref(total_n);
    Eigen::VectorXd U_ref(total_m); 
    for (int k = 0; k < N; ++k) {
        X_ref.segment(k*n, n) = x_ref[k];
    }
    for (int k = 0; k < N; ++k) {
        U_ref.segment(k*m, m) = u_ref[k];
    }
        
    // 计算Hessian矩阵和梯度向量
    Eigen::MatrixXd Fx0 = F * x0;
    H = 2.0 * (Phi.transpose() * Q_total_ * Phi + R_total_dense_); //R_total.toDenseMatrix()也放在初始化中节省时间
    g = 2.0 * (Phi.transpose() * (Q_total_ * (Fx0 - X_ref)) - R_total_ * U_ref);
  
    // 构造约束
    constrains_ = 0;

    // constrain 1: friction cone
    double const mu = 0.5;
    // double const fz_max = 500;
    double const fz_min = 0;
    std::vector<double> const contact_cmd = x0_map.at("contact_cmd");
    int n_friction_cone_constrain = 4*5;
    constrains_ += n_friction_cone_constrain;
    Eigen::MatrixXd friction_matrix_block(5, 3);
    friction_matrix_block << 1,  0, mu,
                            -1,  0, mu,
                             0,  1, mu,
                             0, -1, mu,
                             0,  0, 1;

    Eigen::MatrixXd friction_matrix = Eigen::MatrixXd::Zero(n_friction_cone_constrain, m);
    for (int i=0; i<n_contact_wb; ++i) {
        friction_matrix.block(0+5*i, 6+3*i, 5, 3) = friction_matrix_block;
    }

    std::vector<double> fz_max(4, 0);
    double const fmax = 500;
    fz_max[0] = fmax;
    fz_max[1] = fmax;
    fz_max[2] = fmax;
    fz_max[3] = fmax;

    // if (s_idx == 0) {
    //     fz_max[0] = contact_cmd[0]*fmax;
    //     fz_max[1] = contact_cmd[1]*fmax;
    //     fz_max[2] = contact_cmd[2]*fmax;
    //     fz_max[3] = contact_cmd[3]*fmax;
    // } else {
    //     fz_max[0] = contact_cmd[2]*fmax;
    //     fz_max[1] = contact_cmd[3]*fmax;
    //     fz_max[2] = contact_cmd[0]*fmax;
    //     fz_max[3] = contact_cmd[1]*fmax;
    //     // fz_max[0] = contact_cmd[0]*fmax;
    //     // fz_max[1] = contact_cmd[1]*fmax;
    //     // fz_max[2] = contact_cmd[2]*fmax;
    //     // fz_max[3] = contact_cmd[3]*fmax;
    // }

    Eigen::VectorXd vec_friction_min(n_friction_cone_constrain);
    Eigen::VectorXd vec_friction_max(n_friction_cone_constrain);
    vec_friction_min << 0, 0, 0, 0, fz_min,
                        0, 0, 0, 0, fz_min,
                        0, 0, 0, 0, fz_min,
                        0, 0, 0, 0, fz_min;
    vec_friction_max << fz_max[0],  fz_max[0],  fz_max[0],  fz_max[0],  fz_max[0],
                        fz_max[1],  fz_max[1],  fz_max[1],  fz_max[1],  fz_max[1],
                        fz_max[2],  fz_max[2],  fz_max[2],  fz_max[2],  fz_max[2],
                        fz_max[3],  fz_max[3],  fz_max[3],  fz_max[3],  fz_max[3];

    Eigen::MatrixXd Ac_friction_cone = Eigen::MatrixXd::Zero(n_friction_cone_constrain*N, m*N);
    Eigen::VectorXd lbAc_friction_cone = Eigen::VectorXd::Zero(n_friction_cone_constrain*N);
    Eigen::VectorXd ubAc_friction_cone = Eigen::VectorXd::Zero(n_friction_cone_constrain*N);
    for (int k = 0; k < N; ++k) { 
        Ac_friction_cone.block(k*n_friction_cone_constrain, k*m, n_friction_cone_constrain, m) = friction_matrix;
        lbAc_friction_cone.segment(k*n_friction_cone_constrain, n_friction_cone_constrain) = vec_friction_min;
        ubAc_friction_cone.segment(k*n_friction_cone_constrain, n_friction_cone_constrain) = vec_friction_max;
    }

    // constrain 2: foot noslip
    double const epsilon = 5e-3;
    // double const epsilon_z = 5e-3;

    int n_noslip_constrain = 2*3;
    constrains_ += n_noslip_constrain;

    Eigen::MatrixXd J_matrix = Eigen::MatrixXd::Zero(6, 12);
    J_matrix.block(0, 0, 3, 12) = contact_cmd[s_idx]*quadruped_model_.J_linear_sub_[s_idx];
    J_matrix.block(3, 0, 3, 12) = contact_cmd[s_idx+1]*quadruped_model_.J_linear_sub_[s_idx+1];
    Eigen::MatrixXd J_select = Eigen::MatrixXd::Zero(n_noslip_constrain*N, total_n);
    for (int k = 0; k < N; ++k) {
        J_select.block(n_noslip_constrain*k, n*k+12, n_noslip_constrain, 12) = J_matrix;
    }

    Eigen::VectorXd vec_foot_vel_max = epsilon*Eigen::VectorXd::Ones(n_noslip_constrain*N);
    Eigen::VectorXd vec_foot_vel_min = -vec_foot_vel_max;
    // for(int i=0; i<N; ++i) {
    //     vec_foot_vel_max(i*6+2) = epsilon_z;
    //     vec_foot_vel_max(i*6+5) = epsilon_z;
    // }

    double const inf = std::numeric_limits<double>::infinity();
    Eigen::MatrixXd Ac_noslip = J_select*Phi;
    Eigen::VectorXd vec_foot_vel = J_select*Fx0;
    Eigen::VectorXd lbAc_noslip = vec_foot_vel_min - vec_foot_vel;
    Eigen::VectorXd ubAc_noslip = vec_foot_vel_max - vec_foot_vel;

    // 依次填充子矩阵到对应位置
    Ac = Eigen::MatrixXd::Zero(constrains_*N, m*N);
    lbAc = Eigen::VectorXd::Zero(constrains_*N);
    ubAc = Eigen::VectorXd::Zero(constrains_*N);

    // 方法1构造约束矩阵，更直观 ？？？两种约束方法效果居然不一致？？？
    // for (int k = 0; k < N; ++k) {
    //     Ac.block(k*constrains_, k*m, n_friction_cone_constrain, m) = Ac_friction_cone.block(k*n_friction_cone_constrain, k*m, n_friction_cone_constrain, m);
    //     Ac.block(k*constrains_+n_friction_cone_constrain, k*m, n_noslip_constrain, m) = Ac_noslip.block(k*n_noslip_constrain, k*m, n_noslip_constrain, m);

    //     lbAc.segment(k*constrains_, n_friction_cone_constrain) = lbAc_friction_cone.segment(k*n_friction_cone_constrain, n_friction_cone_constrain);  // 从索引0开始，填充a的2个元素
    //     lbAc.segment(k*constrains_+n_friction_cone_constrain, n_noslip_constrain) = lbAc_noslip.segment(k*n_noslip_constrain, n_noslip_constrain);  // 从索引0开始，填充a的2个元素

    //     ubAc.segment(k*constrains_, n_friction_cone_constrain) = ubAc_friction_cone.segment(k*n_friction_cone_constrain, n_friction_cone_constrain);  // 从索引0开始，填充a的2个元素
    //     ubAc.segment(k*constrains_+n_friction_cone_constrain, n_noslip_constrain) = ubAc_noslip.segment(k*n_noslip_constrain, n_noslip_constrain);  // 从索引0开始，填充a的2个元素
    // }

    // 方法2构造约束矩阵，更直接
    Ac.block(0, 0, n_friction_cone_constrain*N, m*N) = Ac_friction_cone;
    Ac.block(n_friction_cone_constrain*N, 0, n_noslip_constrain*N, m*N) = Ac_noslip;
    lbAc.segment(0, n_friction_cone_constrain*N) = lbAc_friction_cone;  // 从索引0开始，填充a的2个元素
    lbAc.segment(n_friction_cone_constrain*N, n_noslip_constrain*N) = lbAc_noslip;  // 从索引2开始，填充b的3个元素
    ubAc.segment(0, n_friction_cone_constrain*N) = ubAc_friction_cone;  // 从索引0开始，填充a的2个元素
    ubAc.segment(n_friction_cone_constrain*N, n_noslip_constrain*N) = ubAc_noslip;  // 从索引2开始，填充b的3个元素

    return;
}

bool codmpcSolver::qpOASESsolve(Eigen::VectorXd const &x0, std::map<std::string,std::vector<double>> const &x0_map,
                                std::vector<Eigen::VectorXd> const &x_ref,
                                std::vector<Eigen::VectorXd> const &u_ref,
                                std::string const &subsystems_name) {
  
    if(!is_solver_initialized) {
        qpOASESinit();
        is_solver_initialized = true;
        std::cout << "qpOASES initialized!!!" << std::endl;
    }
  
    // 参数设置
    int const &N = config_param_.N_;
    // int const &n = config_param_.n_state;
    int const &m = config_param_.n_control;
    int const c = 6;

    // 计算H矩阵和g向量
    Eigen::MatrixXd H;
    Eigen::VectorXd g;
    Eigen::MatrixXd Ac;
    Eigen::VectorXd lbAc;
    Eigen::VectorXd ubAc;
    computeQPmatrices(subsystems_name, x0, x0_map, x_ref, u_ref, H, g, Ac, lbAc, ubAc);
    
    // 创建qpOASES问题
    qpOASES::SQProblem qp(N*m, N*constrains_);
    // qpOASES::SQProblem qp(N*m, 0);
    
    // 设置求解器选项
    qpOASES::Options options;
    options.setToMPC();
    options.printLevel = qpOASES::PL_NONE;
    // options.printLevel = qpOASES::PL_DEBUG_ITER;
    qp.setOptions(options);
    
    // 初始化问题
    int nWSR = 1000;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> H_R = H; // qpOASES接收的矩阵为行向量！！！
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ac_R = Ac;
    qpOASES::returnValue status = qp.init(H_R.data(), g.data(), Ac_R.data(), nullptr, nullptr, lbAc.data(), ubAc.data(), nWSR);
    
    if (status == qpOASES::SUCCESSFUL_RETURN) {
        // 获取最优解
        Eigen::VectorXd u_opt(N*m);
        qp.getPrimalSolution(u_opt.data());
        
        // 保存控制序列
        for (int i = 0; i < N; ++i) {
            u_[subsystems_name][i] = u_opt.segment(i*m, m);
        }
        
        return true;
    } else {
        std::cerr << "QP solving failed! Error code: " << status << std::endl;
    }
    
    return false;
}

#endif
