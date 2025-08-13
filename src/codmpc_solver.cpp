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

                // set grf and grf_aux //这里和原代码不同，我们只优化grf、grf_aux而不是grf_wb，因此只给当前子系统赋值即可
                counter=0;
                for(auto idx : solver_param_.subsystems_map_contact[problem]) //循环3*2=6次
                {
                    u_ref_k(6+counter) = ref.at("grf")[k][3*idx];
                    u_ref_k(7+counter) = ref.at("grf")[k][3*idx+1];
                    u_ref_k(8+counter) = ref.at("grf")[k][3*idx+2];
                    counter+=3;              
                }
                counter=0;
                for(auto idx : solver_param_.subsystems_map_contact[problem]) //循环3*2=6次
                {

                    u_ref_k(12+counter) = 0.0;
                    u_ref_k(13+counter) = 0.0;
                    u_ref_k(14+counter) = 0.0;
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
            bool success = qpOASESsolve(x0, x_ref, u_ref, problem);
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
        quadruped_model_.updateGrfOld(data_["wb"].grf[0]);
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

// 计算MPC问题转化为QP问题时的H矩阵和g向量
// 使用Eigen::DiagonalMatrix存储Q和R，利用Eigen内部优化
void codmpcSolver::computeQPmatrices(std::string const &subsystems_name,
    Eigen::VectorXd const &x0, 
    std::vector<Eigen::VectorXd> const &x_ref,
    std::vector<Eigen::VectorXd> const &u_ref,
    Eigen::MatrixXd& H, Eigen::VectorXd& g) {
    // 参数设置
    int const &PREDICTION_HORIZON = solver_param_.N_;
    int const &STATE_DIM = solver_param_.n_state;
    int const &CONTROL_DIM = solver_param_.n_control;

    Eigen::MatrixXd const &A_d = quadruped_model_.Ak_[subsystems_name];
    Eigen::MatrixXd const &B_d = quadruped_model_.Bk_[subsystems_name];
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> const &Q = Q_;
    Eigen::DiagonalMatrix<double, Eigen::Dynamic> const &R = R_;

    // 初始化H矩阵和g向量
    const int total_control_dim = PREDICTION_HORIZON * CONTROL_DIM;
    H = Eigen::MatrixXd::Zero(total_control_dim, total_control_dim);
    g = Eigen::VectorXd::Zero(total_control_dim);
    
    // 存储状态预测的中间结果
    std::vector<Eigen::MatrixXd> Phi(PREDICTION_HORIZON);
    std::vector<Eigen::MatrixXd> Gamma(PREDICTION_HORIZON);
    
    // 初始化Phi和Gamma
    Phi[0] = A_d;
    Gamma[0] = B_d;
    
    // 计算后续的Phi和Gamma
    for (int i = 1; i < PREDICTION_HORIZON; ++i) {
        Phi[i] = A_d * Phi[i-1];
        Gamma[i] = A_d * Gamma[i-1];
    }
    
    // 构建H矩阵和g向量
    for (int k = 0; k < PREDICTION_HORIZON; ++k) {
        const int start_idx = k * CONTROL_DIM;
        
        // 计算当前控制输入对应的H矩阵块
        // 利用Eigen对DiagonalMatrix的优化处理矩阵乘法
        Eigen::MatrixXd H_block = R.toDenseMatrix();  // R是对角矩阵
        
        // 添加状态成本对H矩阵的贡献
        for (int i = k; i < PREDICTION_HORIZON; ++i) {
            // 获取当前的Gamma矩阵
            const Eigen::MatrixXd& Gamma_ik = (k == 0) ? Gamma[i] : Gamma[i - k];
            
            // Eigen会自动优化DiagonalMatrix参与的乘法运算
            H_block += Gamma_ik.transpose() * Q * Gamma_ik;
        }
        
        // 将计算好的块放入H矩阵
        H.block(start_idx, start_idx, CONTROL_DIM, CONTROL_DIM) = H_block;
        
        // 计算当前控制输入对应的g向量段
        Eigen::VectorXd g_segment = Eigen::VectorXd::Zero(CONTROL_DIM);
        
        // 控制成本对g向量的贡献 (R是对角矩阵)
        g_segment -= R * u_ref[k];
        
        // 状态成本对g向量的贡献
        for (int i = k; i < PREDICTION_HORIZON; ++i) {
            const Eigen::MatrixXd& Gamma_ik = (k == 0) ? Gamma[i] : Gamma[i - k];
            
            // 计算状态预测
            Eigen::VectorXd x_pred = Phi[i] * x0;
            for (int j = 0; j < i; ++j) {
                x_pred += Gamma[j] * u_ref[j];
            }
            
            // 状态误差项 (x_pred - x_ref[i])
            const Eigen::VectorXd state_error = x_pred - x_ref[i];
            
            // Eigen会自动优化DiagonalMatrix参与的乘法运算
            g_segment += Gamma_ik.transpose() * (Q * state_error);
        }
        
        // 将计算好的段放入g向量
        g.segment(start_idx, CONTROL_DIM) = g_segment;
    }
}

bool codmpcSolver::qpOASESsolve(Eigen::VectorXd const &x0, 
                                std::vector<Eigen::VectorXd> const &x_ref,
                                std::vector<Eigen::VectorXd> const &u_ref,
                                std::string const &subsystems_name) {
  
    // 参数设置
    int const &PREDICTION_HORIZON = solver_param_.N_;
    int const &STATE_DIM = solver_param_.n_state;
    int const &CONTROL_DIM = solver_param_.n_control;

    // 计算H矩阵和g向量
    Eigen::MatrixXd H;
    Eigen::VectorXd g;
    computeQPmatrices(subsystems_name, x0, x_ref, u_ref, H, g);
    
    // 创建qpOASES问题
    qpOASES::SQProblem qp(PREDICTION_HORIZON*CONTROL_DIM, 0);
    
    // 设置求解器选项
    qpOASES::Options options;
    options.setToMPC();
    options.printLevel = qpOASES::PL_NONE;
    qp.setOptions(options);
    
    // 初始化问题
    int nWSR = 1000;
    qpOASES::returnValue status = qp.init(H.data(), g.data(), nullptr, nullptr, nullptr, nullptr, nullptr, nWSR);
    
    if (status == qpOASES::SUCCESSFUL_RETURN) {
        // 获取最优解
        Eigen::VectorXd u_opt(PREDICTION_HORIZON*CONTROL_DIM);
        qp.getPrimalSolution(u_opt.data());
        
        // 保存控制序列
        for (int i = 0; i < PREDICTION_HORIZON; ++i) {
            u_[subsystems_name][i] = u_opt.segment(i*CONTROL_DIM, CONTROL_DIM);
        }
        
        return true;
    }
    
    return false;
}

#endif
