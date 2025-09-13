#include "controllers/dwmpc/codmpc_solver.hpp"

codmpcSolver::codmpcSolver()
{}

void codmpcSolver::init(const parameter &solver_param)
{
    std::cout << "codmpcSolver initialization begins..." << std::endl;

    solver_param_ = solver_param;
    for(auto name : solver_param_.subsystems_name)
    {
        pdata subsystem_data{};
        data_[name] = subsystem_data; //全部初始化为空

        std::vector<Eigen::VectorXd> u0(solver_param_.N_, Eigen::VectorXd::Zero(solver_param_.n_control));
        u_[name] = u0;
    }
    
    Q_ = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(Eigen::VectorXd::Zero(solver_param_.n_state));
    R_ = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(Eigen::VectorXd::Zero(solver_param_.n_control));

    quadruped_model_.modelInit(solver_param);

    std::cout << "codmpcSolver initialized!!!" << std::endl;

#ifdef DEBUG_MODE
    logger_front_.init("front_data.csv", solver_param_.n_state, solver_param_.n_control);
    logger_back_.init("back_data.csv", solver_param_.n_state, solver_param_.n_control);
#endif

    return;
}

void codmpcSolver::solve( bool &do_init,
                    const std::map<std::string,std::vector<double>> &x0_map,
                    const std::map<std::string,std::vector<std::vector<double>>> &ref,
                    const std::map<std::string,std::vector<std::vector<double>>> &param,
                    const std::map<std::string,std::vector<double>> &weight_vec) 
{   
    // init data to the reference 
    int max_iterations{1}; // maximum number of iteration for the distributed solver
    if (do_init)
    {    
        for(auto name : solver_param_.subsystems_name)
        {   
            // init to refernce
            data_[name].p = ref.at("p"); // data_['back'] 或 data_['front'] 放入的是参考轨迹，下面也叫data_[problem] ......
            data_[name].quat = ref.at("quat");
            data_[name].rpy = ref.at("rpy");
            data_[name].dp = ref.at("dp");
            data_[name].omega = ref.at("omega");
            for (int k{0};k<solver_param_.N_+1;k++)
            {   
                std::vector<double> q,dq,tau,grf,foot;
                for (auto idx : solver_param_.subsystems_map_joint[name])
                {
                    q.push_back(ref.at("q")[k][idx]);
                    dq.push_back(ref.at("dq")[k][idx]);
                    tau.push_back(ref.at("tau")[k][idx]);
                }
                for (auto idx : solver_param_.subsystems_map_contact[name])
                {
                    grf.push_back(ref.at("grf")[k][3*idx]);
                    grf.push_back(ref.at("grf")[k][3*idx+1]);
                    grf.push_back(ref.at("grf")[k][3*idx+2]);

                    foot.push_back(ref.at("foot")[k][3*idx]);
                    foot.push_back(ref.at("foot")[k][3*idx+1]);
                    foot.push_back(ref.at("foot")[k][3*idx+2]);
                }
                data_[name].tau.push_back(tau);
                data_[name].grf.push_back(grf);
                data_[name].foot.push_back(foot);
                data_[name].q.push_back(q);
                data_[name].dq.push_back(dq);
                data_[name].dual.push_back(std::vector<double>(6,0));
                data_[name].residual.push_back(std::vector<double>(6,0));
            }
        }
        max_iterations = solver_param_.max_iteration;
        // do_init = false; // 第一次一直为true，直到第二次运行结束后才为false
    } 
    else if (solver_param_.receding_horizon) //第一次不执行，后面执行 难道还可以不执行？？？
    {
        // shift past prediction for receding horizon
        for(auto problem : solver_param_.subsystems_name)
        {
            for (int k{0};k<solver_param_.N_;k++)
            {
                data_[problem].p[k] = data_[problem].p[k+1];
                data_[problem].quat[k] = data_[problem].quat[k+1];
                data_[problem].rpy[k] = data_[problem].rpy[k+1];
                data_[problem].q[k] = data_[problem].q[k+1];
                data_[problem].dp[k] = data_[problem].dp[k+1];
                data_[problem].omega[k] = data_[problem].omega[k+1];
                data_[problem].dq[k] = data_[problem].dq[k+1];          
                if (k < solver_param_.N_ - 1) {
                    data_[problem].tau[k] = data_[problem].tau[k+1];
                    data_[problem].grf[k] = data_[problem].grf[k+1];                
                }
    
                data_[problem].grf[k] = data_[problem].grf[k+1];
                data_[problem].foot[k] = data_[problem].foot[k+1];
                data_[problem].dual[k] = data_[problem].dual[k+1];
            }
            data_[problem].p[solver_param_.N_] = std::vector<double>(3,0);
            data_[problem].quat[solver_param_.N_] = std::vector<double>(4,0);
            data_[problem].rpy[solver_param_.N_] = std::vector<double>(3,0);
            data_[problem].q[solver_param_.N_] = std::vector<double>(solver_param_.subsystems_map_joint[problem].size(),0);
            data_[problem].dp[solver_param_.N_] = std::vector<double>(3,0);
            data_[problem].omega[solver_param_.N_] = std::vector<double>(3,0);
            data_[problem].dq[solver_param_.N_] = std::vector<double>(solver_param_.subsystems_map_joint[problem].size(),0);
            data_[problem].tau[solver_param_.N_-1] = std::vector<double>(solver_param_.subsystems_map_joint[problem].size(),0);
            data_[problem].grf[solver_param_.N_] = std::vector<double>(solver_param_.subsystems_map_contact[problem].size()*3,0);
            data_[problem].foot[solver_param_.N_] = std::vector<double>(solver_param_.subsystems_map_contact[problem].size()*3,0);
            data_[problem].dual[solver_param_.N_] = std::vector<double>(6,0);
        }
        max_iterations = 1; //执行一次又把这个置为1了...好迷的逻辑
    }

    // main loop  (number of iteration)
    for(auto i = 0; i < max_iterations;i++)
    {   
        //problem loop
        for (auto problem : solver_param_.subsystems_name)
        {   
            // if whole body problem skip
            if (problem == "wb")
                continue;

            int counter = 0;

            // ============       MODEL       ============
            quadruped_model_.modelUpdate(x0_map);

            // ============ INITAIAL CONDITION ============
            Eigen::VectorXd x0(solver_param_.n_state);
            
            x0(0) = x0_map.at("p")[0];
            x0(1) = x0_map.at("p")[1];
            x0(2) = x0_map.at("p")[2];

            x0(3) = normalizeAngle(x0_map.at("rpy")[0]);
            x0(4) = normalizeAngle(x0_map.at("rpy")[1]);
            x0(5) = normalizeAngle(x0_map.at("rpy")[2]);
            // problem_initial_condition.push_back(normalizeAngle(x0_map.at("rpy")[0] - ref.at("rpy")[0][0])); // 姿态欧拉角ref设置为0，由于欧拉角有过圈问题，在这里先算好误差
            // problem_initial_condition.push_back(normalizeAngle(x0_map.at("rpy")[1] - ref.at("rpy")[0][1]));
            // problem_initial_condition.push_back(normalizeAngle(x0_map.at("rpy")[2] - ref.at("rpy")[0][2]));

            counter = 0;
            for(auto idx : solver_param_.subsystems_map_joint[problem]) //循环6次
            {
                x0(6+counter) = x0_map.at("q")[idx];
                ++counter;
            }

            x0(12) = x0_map.at("dp")[0];
            x0(13) = x0_map.at("dp")[1];
            x0(14) = x0_map.at("dp")[2];

            x0(15) = x0_map.at("omega")[0];
            x0(16) = x0_map.at("omega")[1];
            x0(17) = x0_map.at("omega")[2];

            counter=0;
            for(auto idx : solver_param_.subsystems_map_joint[problem]) //循环6次
            {
                x0(18+counter) = x0_map.at("dq")[idx];
                ++counter;
            }
            
            counter=0;
            for (auto idx : solver_param_.subsystems_map_contact[problem]) //循环3*2次
            {
                x0(24+counter) = x0_map.at("foot")[3*idx];
                x0(25+counter) = x0_map.at("foot")[3*idx+1];
                x0(26+counter) = x0_map.at("foot")[3*idx+2];
                counter+=3;
            }

            x0(30) = data_["wb"].dp[0][0] - data_[problem].dual[0][0]; //！！！这里给consensus的ref。consensus的ref=barw-y，python里再减去w，即r-y
            x0(31) = data_["wb"].dp[0][1] - data_[problem].dual[0][1];
            x0(32) = data_["wb"].dp[0][2] - data_[problem].dual[0][2];
            x0(33) = data_["wb"].omega[0][0] - data_[problem].dual[0][3];
            x0(34) = data_["wb"].omega[0][1] - data_[problem].dual[0][4];
            x0(35) = data_["wb"].omega[0][2] - data_[problem].dual[0][5];

            x0(36) = 1.0;

            x0_[problem] = x0;

            //std::cout << std::endl;

            ////  ============ REFERENCE  ============
            std::vector<Eigen::VectorXd> x_ref;
            std::vector<Eigen::VectorXd> u_ref;

            // horizon loop
            for (auto k{0};k<solver_param_.N_+1; k++)
            {   
                Eigen::VectorXd x_ref_k(solver_param_.n_state);
    
                //set p,quat 
                x_ref_k(0) = ref.at("p")[k][0];
                x_ref_k(1) = ref.at("p")[k][1];
                x_ref_k(2) = ref.at("p")[k][2];

                x_ref_k(3) = ref.at("rpy")[k][0];
                x_ref_k(4) = ref.at("rpy")[k][1];
                x_ref_k(5) = ref.at("rpy")[k][2];
                // ref_k.push_back(0); // 姿态欧拉角ref设置为0，由于欧拉角有过圈问题，在这里先算好误差
                // ref_k.push_back(0);
                // ref_k.push_back(0);

                // set q
                counter=0;
                for(auto idx : solver_param_.subsystems_map_joint[problem]) //循环6次
                {
                    x_ref_k(6+counter) = ref.at("q")[k][idx];
                    ++counter;
                }

                // set dp omega
                x_ref_k(12) = ref.at("dp")[k][0];
                x_ref_k(13) = ref.at("dp")[k][1];
                x_ref_k(14) = ref.at("dp")[k][2];

                x_ref_k(15) = ref.at("omega")[k][0];
                x_ref_k(16) = ref.at("omega")[k][1];
                x_ref_k(17) = ref.at("omega")[k][2];

                // set dq
                counter=0;
                for(auto idx : solver_param_.subsystems_map_joint[problem]) //循环6次
                {
                    x_ref_k(18+counter) = ref.at("dq")[k][idx];
                    ++counter;
                }
                // set foot
                counter=0;
                for (auto idx : solver_param_.subsystems_map_contact[problem]) //循环2*3次
                {
                    x_ref_k(24+counter) = ref.at("foot")[k][3*idx];
                    x_ref_k(25+counter) = ref.at("foot")[k][3*idx+1];
                    x_ref_k(26+counter) = ref.at("foot")[k][3*idx+2];
                    counter+=3;
                }
                // set consensus
                x_ref_k(30) = data_["wb"].dp[k][0] - data_[problem].dual[k][0]; //！！！这里给consensus的ref。consensus的ref=barw-y，python里再减去w，即r-y
                x_ref_k(31) = data_["wb"].dp[k][1] - data_[problem].dual[k][1];
                x_ref_k(32) = data_["wb"].dp[k][2] - data_[problem].dual[k][2];

                x_ref_k(33) = data_["wb"].omega[k][0] - data_[problem].dual[k][3];
                x_ref_k(34) = data_["wb"].omega[k][1] - data_[problem].dual[k][4];
                x_ref_k(35) = data_["wb"].omega[k][2] - data_[problem].dual[k][5];

                x_ref_k(36) = 1.0;

                x_ref.push_back(x_ref_k);

                ////  ============ REFERENCE  U ============
                Eigen::VectorXd u_ref_k(solver_param_.n_control);
                // set tau
                counter=0;
                for(auto idx : solver_param_.subsystems_map_joint[problem]) //循环6次
                {
                    u_ref_k(counter) = ref.at("tau")[k][idx];
                    ++counter;
                }

                // set grf //这里和原代码不同，我们只优化grf而不是grf_wb，因此只给当前子系统赋值即可
                counter=0;
                for(auto idx : solver_param_.subsystems_map_contact["wb"]) //循环3*2=6次
                {
                    u_ref_k(6+counter) = ref.at("grf")[k][3*idx];
                    u_ref_k(7+counter) = ref.at("grf")[k][3*idx+1];
                    u_ref_k(8+counter) = ref.at("grf")[k][3*idx+2];
                    counter+=3;              
                }

                u_ref.push_back(u_ref_k);

                ////  ============ WEIGHT  ============                
                if(do_init) //本来是每个预测step都有一个权重，这里就不改了
                {
                    // weight p 
                    Q_.diagonal()[0] = weight_vec.at("p")[0];
                    Q_.diagonal()[1] = weight_vec.at("p")[1];
                    Q_.diagonal()[2] = weight_vec.at("p")[2];

                    // weight quat
                    Q_.diagonal()[3] = weight_vec.at("quat")[0];
                    Q_.diagonal()[4] = weight_vec.at("quat")[1];
                    Q_.diagonal()[5] = weight_vec.at("quat")[2];

                    // weight q
                    counter = 0;
                    for(auto idx : solver_param_.subsystems_map_joint[problem]) // 实际循环6次
                    {
                        Q_.diagonal()[6+counter] = weight_vec.at("q")[0];
                        counter++;
                    }
                
                    // weight dp
                    Q_.diagonal()[12] = weight_vec.at("dp")[0];
                    Q_.diagonal()[13] = weight_vec.at("dp")[1];
                    Q_.diagonal()[14] = weight_vec.at("dp")[2];

                    // weight omega
                    Q_.diagonal()[15] = weight_vec.at("omega")[0];
                    Q_.diagonal()[16] = weight_vec.at("omega")[1];
                    Q_.diagonal()[17] = weight_vec.at("omega")[2];

                    // weight dq
                    counter = 0;
                    for(auto idx : solver_param_.subsystems_map_joint[problem]) // 实际循环6次
                    {
                        Q_.diagonal()[18+counter] = weight_vec.at("dq")[0];
                        counter++;
                    }

                    // weight foot
                    counter = 0;
                    for(auto idx : solver_param_.subsystems_map_contact[problem]) // 实际循环2*3次
                    {   
                        if (param.at("contact_seq")[k][idx] == 1)
                        {
                            Q_.diagonal()[24 + counter] = weight_vec.at("foot_stance")[0];
                            Q_.diagonal()[25 + counter] = weight_vec.at("foot_stance")[1];
                            Q_.diagonal()[26 + counter] = weight_vec.at("foot_stance")[2];
                        }
                        else
                        {
                            Q_.diagonal()[24 + counter] = weight_vec.at("foot_swing")[0]; //分配摆动腿或站立腿权重
                            Q_.diagonal()[25 + counter] = weight_vec.at("foot_swing")[1];
                            Q_.diagonal()[26 + counter] = weight_vec.at("foot_swing")[2];
                        }
                        counter+=3;
                    }

                    // weight consensus 
                    Q_.diagonal()[30] = weight_vec.at("consensus")[0];
                    Q_.diagonal()[31] = weight_vec.at("consensus")[0];
                    Q_.diagonal()[32] = weight_vec.at("consensus")[0];
                    Q_.diagonal()[33] = weight_vec.at("consensus")[0];
                    Q_.diagonal()[34] = weight_vec.at("consensus")[0];
                    Q_.diagonal()[35] = weight_vec.at("consensus")[0];

                    // weight constant 1
                    Q_.diagonal()[36] = 0;

                    // weight tau
                    counter = 0;
                    for(auto idx : solver_param_.subsystems_map_joint[problem]) // 实际循环6次
                    {
                        R_.diagonal()[counter] = weight_vec.at("tau")[0];
                        counter++;
                    }

                    // weight grf grf_aux
                    counter = 0;
                    for(auto idx : solver_param_.subsystems_map_contact["wb"])
                    {
                        R_.diagonal()[6+counter] = weight_vec.at("grf")[0];
                        R_.diagonal()[7+counter] = weight_vec.at("grf")[0];                            
                        R_.diagonal()[8+counter] = weight_vec.at("grf")[0];
                        counter+=3;
                    }
                }              
            }
            // pass to the codmpc sovler
            // sendSolverData(problem_ref, x0, data_["wb"].tau[0]);  
            // receiveSolverResult();
#ifdef USE_QPOASES 
            bool success = qpOASESsolve(x0, x0_map, x_ref, u_ref, problem);
            if (!success) {
                std::cout << "MPC求解失败！" << std::endl;
            }
#endif  

#ifdef DEBUG_MODE
            //记录数据
            if (problem == "front") {
                logger_front_.logData(x0, x_ref[0], u_[problem][0], u_ref[0]);
            } else if (problem == "back") {
                logger_back_.logData(x0, x_ref[0], u_[problem][0], u_ref[0]);
            } else {}
#endif
        }     
        for (auto problem : solver_param_.subsystems_name)
        {   
            if (problem == "wb")
                continue;
            // update state from solution
            std::vector<Eigen::VectorXd> x = quadruped_model_.updatePrediction(x0_[problem], u_[problem], problem);
            int n_joints {solver_param_.subsystems_map_joint[problem].size()}; //6
            int counter = 0;
            //update data state
            for (int k{0};k<solver_param_.N_+1;k++)
            {   
                //p 
                data_[problem].p[k][0] = x[k](0);
                data_[problem].p[k][1] = x[k](1);
                data_[problem].p[k][2] = x[k](2);

                //rpy quat
                data_[problem].rpy[k][0] = x[k](3);
                data_[problem].rpy[k][1] = x[k](4);
                data_[problem].rpy[k][2] = x[k](5);

                Eigen::Quaterniond quat = rpyToquat(Eigen::Vector3d(x[k](3), x[k](4), x[k](5)));

                data_[problem].quat[k][0] = quat.x();
                data_[problem].quat[k][1] = quat.y();
                data_[problem].quat[k][2] = quat.z();
                data_[problem].quat[k][3] = quat.w();

                //q
                counter = 0;
                for(auto idx : solver_param_.subsystems_map_joint[problem]) //6个循环
                {
                    data_["wb"].q[k][idx] = x[k](6+counter);  // data_["wb"]放的是当前及预测状态，但是没放全   
                    counter++;
                }
                
                //dp
                data_[problem].dp[k][0] = x[k](12);
                data_[problem].dp[k][1] = x[k](13);
                data_[problem].dp[k][2] = x[k](14);

                //omega
                data_[problem].omega[k][0] = x[k](15);
                data_[problem].omega[k][1] = x[k](16);
                data_[problem].omega[k][2] = x[k](17);

                //dq
                counter = 0;
                for(auto idx : solver_param_.subsystems_map_joint[problem])  //6个循环
                {
                    data_["wb"].dq[k][idx] = x[k](18+counter);
                    counter++;
                }

                //foot
                counter = 0;
                for (auto idx : solver_param_.subsystems_map_contact[problem]) //循环2*3次
                {
                    data_["wb"].foot[k][3*idx]   = x[k](24+counter);
                    data_["wb"].foot[k][3*idx+1] = x[k](25+counter);
                    data_["wb"].foot[k][3*idx+2] = x[k](26+counter);
                    counter += 3;
                }

                // consensus: dp omega 不记录，在下面更新，放在data_["wb"]中而不是data_[problem]中

                //control input
                if (k < solver_param_.N_) {
                    //tau
                    counter = 0;
                    for(auto idx : solver_param_.subsystems_map_joint[problem])
                    {
                        data_["wb"].tau[k][idx] = u_[problem][k](counter);
                        counter++;
                    }

                    //grf
                    counter = 0;
                    for(auto idx : solver_param_.subsystems_map_contact[problem])
                    {
                        data_["wb"].grf[k][3*idx] = u_[problem][k](n_joints+3*counter);
                        data_["wb"].grf[k][3*idx+1] = u_[problem][k](n_joints+3*counter+1);
                        data_["wb"].grf[k][3*idx+2] = u_[problem][k](n_joints+3*counter+2);
                        counter++;
                    }
                }       
            }
        }
        // update whole body speeds
        for(int k{0};k<solver_param_.N_+1;k++)
        {
            data_["wb"].dp[k][0] = 0;
            data_["wb"].dp[k][1] = 0;
            data_["wb"].dp[k][2] = 0;

            data_["wb"].omega[k][0] = 0;
            data_["wb"].omega[k][1] = 0;
            data_["wb"].omega[k][2] = 0;

            double n{solver_param_.subsystems_name.size()-1}; //n=2

            for(auto problem : solver_param_.subsystems_name)
            {
                if (problem == "wb")
                    continue;
                data_["wb"].dp[k][0] += (data_[problem].dp[k][0] + data_[problem].dual[k][0]/weight_vec.at("consensus")[0])/n; // barw_{k+1} = barw_k + (w_k + y_k/rho)/2 来源依据？？？
                data_["wb"].dp[k][1] += (data_[problem].dp[k][1] + data_[problem].dual[k][1]/weight_vec.at("consensus")[0])/n;
                data_["wb"].dp[k][2] += (data_[problem].dp[k][2] + data_[problem].dual[k][2]/weight_vec.at("consensus")[0])/n;

                data_["wb"].omega[k][0] += (data_[problem].omega[k][0] + data_[problem].dual[k][3]/weight_vec.at("consensus")[0])/n;
                data_["wb"].omega[k][1] += (data_[problem].omega[k][1] + data_[problem].dual[k][4]/weight_vec.at("consensus")[0])/n;
                data_["wb"].omega[k][2] += (data_[problem].omega[k][2] + data_[problem].dual[k][5]/weight_vec.at("consensus")[0])/n;
            }
        }
        // update dual
        for(auto problem : solver_param_.subsystems_name)
        {
            if (problem == "wb")
            {
                continue;
            }
            for (int k{0};k<solver_param_.N_;k++)
            {
                data_[problem].dual[k][0] += (data_[problem].dp[k][0] - data_["wb"].dp[k][0])*weight_vec.at("consensus")[0]; // y_{k+1} = y_k + (w_k - barw_{k+1})*rho 和公式(3)一致
                data_[problem].dual[k][1] += (data_[problem].dp[k][1] - data_["wb"].dp[k][1])*weight_vec.at("consensus")[0];
                data_[problem].dual[k][2] += (data_[problem].dp[k][2] - data_["wb"].dp[k][2])*weight_vec.at("consensus")[0];
                data_[problem].dual[k][3] += (data_[problem].omega[k][0] - data_["wb"].omega[k][0])*weight_vec.at("consensus")[0];
                data_[problem].dual[k][4] += (data_[problem].omega[k][1] - data_["wb"].omega[k][1])*weight_vec.at("consensus")[0];
                data_[problem].dual[k][5] += (data_[problem].omega[k][2] - data_["wb"].omega[k][2])*weight_vec.at("consensus")[0];

                data_[problem].residual[k][0] = (data_[problem].dp[k][0] - data_["wb"].dp[k][0]); // 残差r=w-barw  是不是写反了？？？
                data_[problem].residual[k][1] = (data_[problem].dp[k][1] - data_["wb"].dp[k][1]);
                data_[problem].residual[k][2] = (data_[problem].dp[k][2] - data_["wb"].dp[k][2]);
                data_[problem].residual[k][3] = (data_[problem].omega[k][0] - data_["wb"].omega[k][0]);
                data_[problem].residual[k][4] = (data_[problem].omega[k][1] - data_["wb"].omega[k][1]);
                data_[problem].residual[k][5] = (data_[problem].omega[k][2] - data_["wb"].omega[k][2]);
            }
            // for (int k{0};k<solver_param_.N_;k++)
            // {
            //     data_[problem].dual[k][0] = (data_["front"].dp[k][0] - (data_["back"].dp[k][0]))*weight_vec.at("consensus")[0];
            //     data_[problem].dual[k][1] = (data_["front"].dp[k][1] - (data_["back"].dp[k][1]))*weight_vec.at("consensus")[0];
            //     data_[problem].dual[k][2] = (data_["front"].dp[k][2] - (data_["back"].dp[k][2]))*weight_vec.at("consensus")[0];
            //     data_[problem].dual[k][3] = (data_["front"].omega[k][0] - (data_["back"].omega[k][0]))*weight_vec.at("consensus")[0];
            //     data_[problem].dual[k][4] = (data_["front"].omega[k][1] - (data_["back"].omega[k][1]))*weight_vec.at("consensus")[0];
            //     data_[problem].dual[k][5] = (data_["front"].omega[k][2] - (data_["back"].omega[k][2]))*weight_vec.at("consensus")[0];
            // }
        }
        // check stopping criteria
        //TODO
    }
    do_init = false; // 只有第一次运行为true，后面都为false
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

// void codmpcSolver::sendSolverData(std::vector<std::vector<double>> const &reference, std::vector<double> const &initial_condition, std::vector<double> const &u0_init) {

//      // TODO
//     return;
// }


// void codmpcSolver::receiveSolverResult() {

//      // TODO
//     return;
// }

#ifdef USE_QPOASES

// 构建总权重矩阵 (Q_total和R_total)，预先计算，只算一次
void codmpcSolver::buildTotalWeightMatrices() {

    int const &N = solver_param_.N_;
    int const &n = solver_param_.n_state;
    int const &m = solver_param_.n_control;

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

    int const &N = solver_param_.N_;
    int const &n = solver_param_.n_state;

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

    int const &N = solver_param_.N_;
    int const &n = solver_param_.n_state;
    int const &m = solver_param_.n_control;

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

    // int const &N = solver_param_.N_;
    // int const &n = solver_param_.n_state;
    // int const &m = solver_param_.n_control;

    //构造大权重矩阵
    buildTotalWeightMatrices();

    // 提前设置输入约束
    // constrain 1: friction cone
    // n_friction_cone_constrain_ = 10;
    // double const mu = 0.5;
    // double const fz_max = 500;
    // Eigen::MatrixXd friction_matrix(5, 3);
    // friction_matrix << 1,  0, mu,
    //                   -1,  0, mu,
    //                    0,  1, mu,
    //                    0, -1, mu,
    //                    0,  0,  1;  
    // Eigen::MatrixXd friction_matrix_total = Eigen::MatrixXd::Zero(n_friction_cone_constrain_, m);
    // friction_matrix_total.block(0, 6, 5, 3) = friction_matrix;
    // friction_matrix_total.block(5, 9, 5, 3) = friction_matrix;
    // Ac_friction_cone_ = Eigen::MatrixXd::Zero(n_friction_cone_constrain_*N, m*N);
    // for (int k = 0; k < N; ++k) { 
    //     Ac_friction_cone_.block(k*n_friction_cone_constrain_, k*m, n_friction_cone_constrain_, m) = friction_matrix_total;
    // }
    // Eigen::VectorXd vec_friction_min(n_friction_cone_constrain_);
    // // Eigen::VectorXd vec_friction_max(n_friction_cone_constrain_);
    // double const inf = std::numeric_limits<double>::infinity();
    // vec_friction_min  << 0, 0, 0, 0, 0,
    //                      0, 0, 0, 0, 0;
    // vec_friction_max_ = Eigen::VectorXd::Zero(n_friction_cone_constrain_);
    // vec_friction_max_ << fz_max, fz_max, fz_max, fz_max, fz_max,
    //                      fz_max, fz_max, fz_max, fz_max, fz_max;

    // lbAc_friction_cone_ = Eigen::VectorXd::Zero(n_friction_cone_constrain_*N);
    // ubAc_friction_cone_ = Eigen::VectorXd::Zero(n_friction_cone_constrain_*N);
    // for (int k = 0; k < N; ++k) { 
    //     lbAc_friction_cone_.segment(k*n_friction_cone_constrain_, n_friction_cone_constrain_) = vec_friction_min;
    //     ubAc_friction_cone_.segment(k*n_friction_cone_constrain_, n_friction_cone_constrain_) = vec_friction_max_;
    // }
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
    int const &N = solver_param_.N_;
    int const &n = solver_param_.n_state;
    int const &m = solver_param_.n_control;
    int const &n_contact_wb = solver_param_.n_contact_wb;
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
    double const fz_max = 500;
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

    Eigen::MatrixXd Ac_friction_cone = Eigen::MatrixXd::Zero(n_friction_cone_constrain*N, m*N);
    Eigen::VectorXd lbAc_friction_cone = Eigen::VectorXd::Zero(n_friction_cone_constrain*N);
    Eigen::VectorXd ubAc_friction_cone = Eigen::VectorXd::Zero(n_friction_cone_constrain*N);
    for (int k = 0; k < N; ++k) { 
        Ac_friction_cone.block(k*n_friction_cone_constrain, k*m, n_friction_cone_constrain, m) = friction_matrix;
        lbAc_friction_cone.segment(k*n_friction_cone_constrain, n_friction_cone_constrain) = vec_friction_min;
        ubAc_friction_cone.segment(k*n_friction_cone_constrain, n_friction_cone_constrain) = vec_friction_max;
    }

    // constrain 2: foot noslip
    double const epsilon = 1e-6;

    int n_noslip_constrain = 2*3;
    constrains_ += n_noslip_constrain;
    Eigen::MatrixXd J_matrix = Eigen::MatrixXd::Zero(6, 12);
    J_matrix.block(0, 0, 3, 12) = contact_cmd[s_idx]*quadruped_model_.J_linear_[s_idx];
    J_matrix.block(3, 0, 3, 12) = contact_cmd[s_idx+1]*quadruped_model_.J_linear_[s_idx+1];

    Eigen::MatrixXd J_select = Eigen::MatrixXd::Zero(n_noslip_constrain*N, total_n);
    for (int k = 0; k < N; ++k) {
        J_select.block(n_noslip_constrain*k, n*k+12, n_noslip_constrain, 12) = J_matrix;
    }
    Eigen::VectorXd const vec_foot_vel_max = epsilon*Eigen::VectorXd::Ones(n_noslip_constrain*N);
    double const inf = std::numeric_limits<double>::infinity();
    Eigen::MatrixXd Ac_noslip = J_select*Phi;
    Eigen::VectorXd vec_foot_vel = J_select*Fx0;
    Eigen::VectorXd lbAc_noslip = -vec_foot_vel_max - vec_foot_vel;
    Eigen::VectorXd ubAc_noslip = vec_foot_vel_max - vec_foot_vel;

    // 依次填充子矩阵到对应位置
    Ac = Eigen::MatrixXd::Zero(constrains_*N, m*N);
    lbAc = Eigen::VectorXd::Zero(constrains_*N);
    ubAc = Eigen::VectorXd::Zero(constrains_*N);

    // 方法1构造约束矩阵，更直观
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
  
    if(!is_initialized) {
        qpOASESinit();
        is_initialized = true;
        std::cout << "qpOASES initialized!!!" << std::endl;
    }
  
    // 参数设置
    int const &N = solver_param_.N_;
    // int const &n = solver_param_.n_state;
    int const &m = solver_param_.n_control;
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
