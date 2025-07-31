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

        std::vector<std::vector<double>> u0(solver_param_.N_, std::vector<double>(6, 0.0));
        u_[name] = u0;
    }

    Q_ = std::vector<double>(37, 0.0);
    R_ = std::vector<double>(6,  0.0);

    quadruped_model_.modelInit(solver_param);

    std::cout << "codmpcSolver initialized!!!" << std::endl;

    return;
}

void codmpcSolver::solve( bool &do_init,
                    const std::map<std::string,std::vector<double>> &initial_condition,
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

            // ============       MODEL       ============
            quadruped_model_.modelUpdate(initial_condition);

            // ============ INITAIAL CONDITION ============
            
            std::vector<double> problem_initial_condition;
            
            problem_initial_condition.push_back(initial_condition.at("p")[0]);
            problem_initial_condition.push_back(initial_condition.at("p")[1]);
            problem_initial_condition.push_back(initial_condition.at("p")[2]);

            problem_initial_condition.push_back(normalizeAngle(initial_condition.at("rpy")[0]));
            problem_initial_condition.push_back(normalizeAngle(initial_condition.at("rpy")[1]));
            problem_initial_condition.push_back(normalizeAngle(initial_condition.at("rpy")[2]));
            // problem_initial_condition.push_back(normalizeAngle(initial_condition.at("rpy")[0] - ref.at("rpy")[0][0])); // 姿态欧拉角ref设置为0，由于欧拉角有过圈问题，在这里先算好误差
            // problem_initial_condition.push_back(normalizeAngle(initial_condition.at("rpy")[1] - ref.at("rpy")[0][1]));
            // problem_initial_condition.push_back(normalizeAngle(initial_condition.at("rpy")[2] - ref.at("rpy")[0][2]));

            for(auto idx : solver_param_.subsystems_map_joint[problem]) //循环6次
            {
                problem_initial_condition.push_back(initial_condition.at("q")[idx]);
            }

            problem_initial_condition.push_back(initial_condition.at("dp")[0]);
            problem_initial_condition.push_back(initial_condition.at("dp")[1]);
            problem_initial_condition.push_back(initial_condition.at("dp")[2]);

            problem_initial_condition.push_back(initial_condition.at("omega")[0]);
            problem_initial_condition.push_back(initial_condition.at("omega")[1]);
            problem_initial_condition.push_back(initial_condition.at("omega")[2]);

            for(auto idx : solver_param_.subsystems_map_joint[problem]) //循环6次
            {
                problem_initial_condition.push_back(initial_condition.at("dq")[idx]);
            }
            
            for (auto idx : solver_param_.subsystems_map_contact[problem]) //循环2*3次
            {
                problem_initial_condition.push_back(initial_condition.at("foot")[3*idx]);
                problem_initial_condition.push_back(initial_condition.at("foot")[3*idx+1]);
                problem_initial_condition.push_back(initial_condition.at("foot")[3*idx+2]);
            }

            problem_initial_condition.push_back(data_["wb"].dp[0][0] - data_[problem].dual[0][0]); //！！！这里给consensus的ref。consensus的ref=barw-y，python里再减去w，即r-y
            problem_initial_condition.push_back(data_["wb"].dp[0][1] - data_[problem].dual[0][1]);
            problem_initial_condition.push_back(data_["wb"].dp[0][2] - data_[problem].dual[0][2]);
            problem_initial_condition.push_back(data_["wb"].omega[0][0] - data_[problem].dual[0][3]);
            problem_initial_condition.push_back(data_["wb"].omega[0][1] - data_[problem].dual[0][4]);
            problem_initial_condition.push_back(data_["wb"].omega[0][2] - data_[problem].dual[0][5]);

            problem_initial_condition.push_back(1.0);

            x0_[problem] = problem_initial_condition;

            //std::cout << std::endl;

            std::vector<std::vector<double>> problem_ref;
            std::vector<std::vector<double>> problem_param;
            std::vector<std::vector<double>> problem_weight;
            std::vector<std::vector<double>> problem_constraints;

            // horizon loop
            for (auto k{0};k<solver_param_.N_+1; k++)
            {   
                ////  ============ REFERENCE  ============
                std::vector<double> ref_k;
                
                //set p,quat 
                ref_k.push_back(ref.at("p")[k][0]);
                ref_k.push_back(ref.at("p")[k][1]);
                ref_k.push_back(ref.at("p")[k][2]);

                ref_k.push_back(ref.at("rpy")[k][0]);
                ref_k.push_back(ref.at("rpy")[k][1]);
                ref_k.push_back(ref.at("rpy")[k][2]);
                // ref_k.push_back(0); // 姿态欧拉角ref设置为0，由于欧拉角有过圈问题，在这里先算好误差
                // ref_k.push_back(0);
                // ref_k.push_back(0);

                // set q
                for(auto idx : solver_param_.subsystems_map_joint[problem]) //循环6次
                {   
                    ref_k.push_back(ref.at("q")[k][idx]);
                }

                // set dp omega
                ref_k.push_back(ref.at("dp")[k][0]);
                ref_k.push_back(ref.at("dp")[k][1]);
                ref_k.push_back(ref.at("dp")[k][2]);

                ref_k.push_back(ref.at("omega")[k][0]);
                ref_k.push_back(ref.at("omega")[k][1]);
                ref_k.push_back(ref.at("omega")[k][2]);
                // set dq
                for(auto idx : solver_param_.subsystems_map_joint[problem]) //循环6次
                {
                    ref_k.push_back(ref.at("dq")[k][idx]);
                }
                // set foot
                for (auto idx : solver_param_.subsystems_map_contact[problem]) //循环2*3次
                {
                    ref_k.push_back(ref.at("foot")[k][3*idx]);
                    ref_k.push_back(ref.at("foot")[k][3*idx+1]);
                    ref_k.push_back(ref.at("foot")[k][3*idx+2]);
                }
                // set consensus
                ref_k.push_back(data_["wb"].dp[k][0] - data_[problem].dual[k][0]); //！！！这里给consensus的ref。consensus的ref=barw-y，python里再减去w，即r-y
                ref_k.push_back(data_["wb"].dp[k][1] - data_[problem].dual[k][1]);
                ref_k.push_back(data_["wb"].dp[k][2] - data_[problem].dual[k][2]);

                ref_k.push_back(data_["wb"].omega[k][0] - data_[problem].dual[k][3]);
                ref_k.push_back(data_["wb"].omega[k][1] - data_[problem].dual[k][4]);
                ref_k.push_back(data_["wb"].omega[k][2] - data_[problem].dual[k][5]);

                ref_k.push_back(1.0);

                problem_ref.push_back(ref_k);

                ////  ============ WEIGHT  ============                
                if(do_init) //本来是每个预测step都有一个权重，这里就不改了
                {
                    // weight p 
                    Q_[0] = weight_vec.at("p")[0];
                    Q_[1] = weight_vec.at("p")[1];
                    Q_[2] = weight_vec.at("p")[2];

                    // weight quat
                    Q_[3] = weight_vec.at("quat")[0];
                    Q_[4] = weight_vec.at("quat")[1];
                    Q_[5] = weight_vec.at("quat")[2];

                    // weight q
                    int i = 0;
                    for(auto idx : solver_param_.subsystems_map_joint[problem]) // 实际循环6次
                    {
                        Q_[6+i] = weight_vec.at("q")[0];
                        i++;
                    }
                
                    // weight dp
                    Q_[12] = weight_vec.at("dp")[0];
                    Q_[13] = weight_vec.at("dp")[1];
                    Q_[14] = weight_vec.at("dp")[2];

                    // weight omega
                    Q_[15] = weight_vec.at("omega")[0];
                    Q_[16] = weight_vec.at("omega")[1];
                    Q_[17] = weight_vec.at("omega")[2];

                    // weight dq
                    i = 0;
                    for(auto idx : solver_param_.subsystems_map_joint[problem]) // 实际循环6次
                    {
                        Q_[18+i] = weight_vec.at("dq")[0];
                        i++;
                    }

                    // weight foot
                    i = 0;
                    for(auto idx : solver_param_.subsystems_map_contact[problem]) // 实际循环2*3次
                    {   
                        if (param.at("contact_seq")[k][idx] == 1)
                        {
                            Q_[24 + i] = weight_vec.at("foot_stance")[0];
                            Q_[25 + i] = weight_vec.at("foot_stance")[1];
                            Q_[26 + i] = weight_vec.at("foot_stance")[2];
                        }
                        else
                        {
                            Q_[24 + i] = weight_vec.at("foot_swing")[0]; //分配摆动腿或站立腿权重
                            Q_[25 + i] = weight_vec.at("foot_swing")[1];
                            Q_[26 + i] = weight_vec.at("foot_swing")[2];
                        }
                        i+=3;
                    }

                    // weight consensus 
                    Q_[30] = weight_vec.at("consensus")[0];
                    Q_[31] = weight_vec.at("consensus")[0];
                    Q_[32] = weight_vec.at("consensus")[0];
                    Q_[33] = weight_vec.at("consensus")[0];
                    Q_[34] = weight_vec.at("consensus")[0];
                    Q_[35] = weight_vec.at("consensus")[0];

                    // weight constant 1
                    Q_[36] = 0;

                    // weight tau
                    i = 0;
                    for(auto idx : solver_param_.subsystems_map_joint[problem]) // 实际循环6次
                    {
                        R_[i] = weight_vec.at("tau")[0];
                        i++;
                    }
                }              
            }
            // pass to the codmpc sovler
            sendSolverData(problem_ref, problem_initial_condition, data_["wb"].tau[0]);  
            receiveSolverResult();
        }     

        for (auto problem : solver_param_.subsystems_name)
        {   

            // update state from solution
            std::vector<std::vector<double>> x{};
            if (problem == "wb")
                continue;
            x = quadruped_model_.updatePrediction(x0_[problem], u_[problem], problem);
            int n_joints {solver_param_.subsystems_map_joint[problem].size()}; //6
            int counter{0};
            //update data state
            for (int k{0};k<solver_param_.N_+1;k++)
            {   
                //p 
                data_[problem].p[k][0] = x[k][0];
                data_[problem].p[k][1] = x[k][1];
                data_[problem].p[k][2] = x[k][2];

                //rpy quat
                data_[problem].rpy[k][0] = x[k][3];
                data_[problem].rpy[k][1] = x[k][4];
                data_[problem].rpy[k][2] = x[k][5];

                Eigen::Quaterniond quat = rpyToquat(Eigen::Vector3d(x[k][3], x[k][4], x[k][5]));

                data_[problem].quat[k][0] = quat.x();
                data_[problem].quat[k][1] = quat.y();
                data_[problem].quat[k][2] = quat.z();
                data_[problem].quat[k][3] = quat.w();

                //q
                counter = 0;
                for(auto idx : solver_param_.subsystems_map_joint[problem]) //6个循环
                {
                    data_["wb"].q[k][idx] = x[k][6+counter];  // data_["wb"]放的是当前及预测状态，但是没放全   
                    counter++;
                }
                
                //dp
                data_[problem].dp[k][0] = x[k][12];
                data_[problem].dp[k][1] = x[k][13];
                data_[problem].dp[k][2] = x[k][14];

                //omega
                data_[problem].omega[k][0] = x[k][15];
                data_[problem].omega[k][1] = x[k][16];
                data_[problem].omega[k][2] = x[k][17];

                //dq
                counter = 0;
                for(auto idx : solver_param_.subsystems_map_joint[problem])  //6个循环
                {
                    data_["wb"].dq[k][idx] = x[k][18+counter];
                    counter++;
                }

                //foot
                counter = 0;
                for (auto idx : solver_param_.subsystems_map_contact[problem]) //循环2*3次
                {
                    data_["wb"].foot[k][3*idx]   = x[k][24+counter];
                    data_["wb"].foot[k][3*idx+1] = x[k][25+counter];
                    data_["wb"].foot[k][3*idx+2] = x[k][26+counter];
                    counter += 3;
                }

                // consensus: dp omega 不记录，在下面更新，放在data_["wb"]中而不是data_[problem]中

                //tau
                if (k < solver_param_.N_) {
                    counter = 0;
                    for(auto idx : solver_param_.subsystems_map_joint[problem])
                    {
                        data_["wb"].tau[k][idx] = u_[problem][k][counter];
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

void codmpcSolver::sendSolverData(std::vector<std::vector<double>> const &reference, std::vector<double> const &initial_condition, std::vector<double> const &u0_init) {

     // TODO
    return;
}


void codmpcSolver::receiveSolverResult() {

     // TODO
    return;
}

