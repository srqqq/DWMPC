#include "controllers/dwmpc/codmpc_solver.hpp"
#include <future>
#include <numeric>

namespace
{
template <typename Derived>
std::vector<double> toStdVector(const Eigen::MatrixBase<Derived> &value)
{
    return std::vector<double>(value.derived().data(), value.derived().data() + value.size());
}

hpipm::OcpQpIpmSolverSettings solverSettings()
{
    hpipm::OcpQpIpmSolverSettings settings;
    settings.mode = hpipm::HpipmMode::SpeedAbs;
    settings.iter_max = 100;
    settings.alpha_min = 1e-8;
    settings.mu0 = 1e2;
    settings.tol_stat = 1e-04;
    settings.tol_eq = 1e-04;
    settings.tol_ineq = 1e-04;
    settings.tol_comp = 1e-04;
    settings.reg_prim = 1e-12;
    settings.warm_start = 0;
    settings.pred_corr = 1;
    settings.ric_alg = 1;
    settings.split_step = 1;
    return settings;
}
}

codmpcSolver::codmpcSolver()
{}

codmpcSolver::~codmpcSolver() = default;

void codmpcSolver::init(const parameter &config_param)
{
    std::cout << "codmpcSolver initialization begins..." << std::endl;

    config_param_ = config_param;
    for(auto problem : config_param_.subsystems_name)
    {
        pdata subsystem_data{};
        data_[problem] = subsystem_data; //全部初始化为空

        if (problem != "wb") {
            std::vector<Eigen::VectorXd> u0(config_param_.N_, Eigen::VectorXd::Zero(config_param_.n_control));
            u_[problem] = u0;
            u_ref_[problem] = u0;

            std::vector<Eigen::VectorXd> x0(config_param_.N_+1, Eigen::VectorXd::Zero(config_param_.n_state));
            x_[problem] = x0;
            x_ref_[problem] = x0;
            consensus_ref_[problem] = x0;

            x0_[problem] = Eigen::VectorXd::Zero(config_param_.n_state);
        }
    }
    
    for (std::size_t i = 0; i < 2; ++i)
    {
        Q_[i] = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(Eigen::VectorXd::Zero(config_param_.n_state));
        Q_consensus_[i] = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(Eigen::VectorXd::Zero(config_param_.n_state));
        R_[i] = Eigen::DiagonalMatrix<double, Eigen::Dynamic>(Eigen::VectorXd::Zero(config_param_.n_control));
        qp_[i].resize(config_param_.N_ + 1);
        solution_[i].resize(config_param_.N_ + 1);
        for (int k = 0; k <= config_param_.N_; ++k)
        {
            auto &stage = qp_[i][k];
            stage.Q = Eigen::MatrixXd::Zero(config_param_.n_state, config_param_.n_state);
            stage.q = Eigen::VectorXd::Zero(config_param_.n_state);
            const int num_general_constraints = k == 0 ? 20 : (k == config_param_.N_ ? 6 : 26);
            stage.C = Eigen::MatrixXd::Zero(num_general_constraints, config_param_.n_state);
            stage.D = Eigen::MatrixXd::Zero(num_general_constraints, config_param_.n_control);
            stage.lg = Eigen::VectorXd::Zero(num_general_constraints);
            stage.ug = Eigen::VectorXd::Zero(num_general_constraints);
            if (k < config_param_.N_)
            {
                stage.A = Eigen::MatrixXd::Zero(config_param_.n_state, config_param_.n_state);
                stage.B = Eigen::MatrixXd::Zero(config_param_.n_state, config_param_.n_control);
                stage.b = Eigen::VectorXd::Zero(config_param_.n_state);
                stage.R = Eigen::MatrixXd::Zero(config_param_.n_control, config_param_.n_control);
                stage.S = Eigen::MatrixXd::Zero(config_param_.n_control, config_param_.n_state);
                stage.r = Eigen::VectorXd::Zero(config_param_.n_control);
                stage.idxbu.resize(config_param_.n_joint);
                std::iota(stage.idxbu.begin(), stage.idxbu.end(), 0);
                stage.lbu = Eigen::VectorXd::Zero(config_param_.n_joint);
                stage.ubu = Eigen::VectorXd::Zero(config_param_.n_joint);
            }
        }
        hpipm_solver_[i].setSolverSettings(solverSettings());
        hpipm_solver_[i].resize(qp_[i]);
    }

    quadruped_model_.modelInit(config_param);

    std::cout << "codmpcSolver initialized!!!" << std::endl;

    return;
}

bool codmpcSolver::solve(bool &do_init,
                         const RobotState &state,
                         const ReferenceTrajectory &reference,
                         const MpcWeights &weights)
{   
    auto data_candidate = data_;

    // init data to the reference 
    if (do_init)
    {    
        for(auto problem : config_param_.subsystems_name) //对所有subsystem初始化
        {   
            pdata initialized_data{};
            // init to refernce
            for (int k{0};k<config_param_.N_+1;k++)
            {   
                std::vector<double> q,dq,tau,grf,foot;
                initialized_data.p.push_back(toStdVector(reference.position[k]));
                initialized_data.rpy.push_back(toStdVector(reference.rpy[k]));
                const Eigen::Quaterniond quat = rpyToquat(reference.rpy[k]);
                initialized_data.quat.push_back({quat.x(), quat.y(), quat.z(), quat.w()});
                initialized_data.dp.push_back(toStdVector(reference.linear_velocity[k]));
                const Eigen::Vector3d &ref_rpy = reference.rpy[k];
                const Eigen::Vector3d &ref_omega = reference.angular_velocity[k];
                const Eigen::Vector3d ref_ypr_rate = worldOmegaToYprRate(ref_rpy, ref_omega);
                initialized_data.omega.push_back(
                    {ref_ypr_rate[2], ref_ypr_rate[1], ref_ypr_rate[0]});
                for (auto idx : config_param_.subsystems_map_joint[problem])
                {
                    q.push_back(reference.joint_position[k][idx]);
                    dq.push_back(reference.joint_velocity[k][idx]);
                    tau.push_back(reference.torque[k][idx]);
                }
                for (auto idx : config_param_.subsystems_map_contact[problem])
                {
                    grf.push_back(reference.ground_reaction_force[k][3*idx]);
                    grf.push_back(reference.ground_reaction_force[k][3*idx+1]);
                    grf.push_back(reference.ground_reaction_force[k][3*idx+2]);

                    foot.push_back(reference.foot_position[k][3*idx]);
                    foot.push_back(reference.foot_position[k][3*idx+1]);
                    foot.push_back(reference.foot_position[k][3*idx+2]);
                }
                if (k < config_param_.N_)
                {
                    initialized_data.tau.push_back(tau);
                    initialized_data.grf.push_back(grf);
                }
                initialized_data.foot.push_back(foot);
                initialized_data.q.push_back(q);
                initialized_data.dq.push_back(dq);
                initialized_data.dual.push_back(std::vector<double>(6,0));
                initialized_data.residual.push_back(std::vector<double>(6,0));
            }
            data_candidate[problem] = std::move(initialized_data);
        }
    }

    const double rho = weights.consensus;
    if (rho <= 0.0)
    {
        throw std::invalid_argument("consensus weight must be positive");
    }

    // Retain the ADMM state at the same relative horizon node across control
    // cycles. HPIPM itself remains cold-started.
    auto consensus_dp = data_candidate.at("wb").dp;
    auto consensus_omega = data_candidate.at("wb").omega;
    std::map<std::string, std::vector<std::vector<double>>> dual_for_iteration;
    for (const auto &problem : config_param_.subsystems_name)
    {
        if (problem != "wb")
        {
            dual_for_iteration[problem] = data_candidate.at(problem).dual;
        }
    }

    auto x_candidate = x_;
    auto u_candidate = u_;
    bool all_solved = true;

    // main loop  (number of iteration)
    //problem loop
    // ============       MODEL       ============
    quadruped_model_.modelUpdate(state);

    for (auto problem : config_param_.subsystems_name)
    {   
        // if whole body problem skip
        if (problem == "wb")
            continue;

        const std::size_t solver_index = problem == "front" ? 0 : 1;
        auto &Q = Q_[solver_index];
        auto &Q_consensus = Q_consensus_[solver_index];
        auto &R = R_[solver_index];

        int counter = 0;

        // ============ INITAIAL CONDITION ============ 
        x0_[problem](0) = state.position[0];
        x0_[problem](1) = state.position[1];
        x0_[problem](2) = state.position[2];

        x0_[problem](3) = normalizeAngle(state.rpy[2]);
        x0_[problem](4) = normalizeAngle(state.rpy[1]);
        x0_[problem](5) = normalizeAngle(state.rpy[0]);
        // problem_initial_condition.push_back(normalizeAngle(state.rpy[0] - reference.rpy[0][0])); // 姿态欧拉角ref设置为0，由于欧拉角有过圈问题，在这里先算好误差
        // problem_initial_condition.push_back(normalizeAngle(state.rpy[1] - reference.rpy[0][1]));
        // problem_initial_condition.push_back(normalizeAngle(state.rpy[2] - reference.rpy[0][2]));

        counter = 0;
        for(auto idx : config_param_.subsystems_map_joint[problem]) //循环6次
        {
            x0_[problem](6+counter) = state.joint_position[idx];
            ++counter;
        }

        x0_[problem](12) = state.linear_velocity[0];
        x0_[problem](13) = state.linear_velocity[1];
        x0_[problem](14) = state.linear_velocity[2];

        const Eigen::Vector3d rpy = Eigen::Map<const Eigen::Vector3d>(state.rpy.data());
        const Eigen::Vector3d omega = Eigen::Map<const Eigen::Vector3d>(state.angular_velocity.data());
        x0_[problem].segment<3>(15) = worldOmegaToYprRate(rpy, omega);

        counter=0;
        for(auto idx : config_param_.subsystems_map_joint[problem]) //循环6次
        {
            x0_[problem](18+counter) = state.joint_velocity[idx];
            ++counter;
        }
        
        counter=0;
        for (auto idx : config_param_.subsystems_map_contact[problem]) //循环3*2次
        {
            x0_[problem](24+counter) = state.foot_position[3*idx];
            x0_[problem](25+counter) = state.foot_position[3*idx+1];
            x0_[problem](26+counter) = state.foot_position[3*idx+2];
            counter+=3;
        }

        //std::cout << std::endl;

        ////  ============ REFERENCE  ============
        // horizon loop
        for (auto k{0};k<config_param_.N_+1; k++)
        {   
            //set p,quat 
            x_ref_[problem][k](0) = reference.position[k][0];
            x_ref_[problem][k](1) = reference.position[k][1];
            x_ref_[problem][k](2) = reference.position[k][2];

            x_ref_[problem][k](3) = reference.rpy[k][2];
            x_ref_[problem][k](4) = reference.rpy[k][1];
            x_ref_[problem][k](5) = reference.rpy[k][0];
            // ref_k.push_back(0); // 姿态欧拉角ref设置为0，由于欧拉角有过圈问题，在这里先算好误差
            // ref_k.push_back(0);
            // ref_k.push_back(0);

            // set q
            counter=0;
            for(auto idx : config_param_.subsystems_map_joint[problem]) //循环6次
            {
                x_ref_[problem][k](6+counter) = reference.joint_position[k][idx];
                ++counter;
            }

            // set dp omega
            x_ref_[problem][k](12) = reference.linear_velocity[k][0];
            x_ref_[problem][k](13) = reference.linear_velocity[k][1];
            x_ref_[problem][k](14) = reference.linear_velocity[k][2];

            const Eigen::Vector3d ref_rpy = Eigen::Map<const Eigen::Vector3d>(reference.rpy[k].data());
            const Eigen::Vector3d ref_omega = Eigen::Map<const Eigen::Vector3d>(reference.angular_velocity[k].data());
            x_ref_[problem][k].segment<3>(15) = worldOmegaToYprRate(ref_rpy, ref_omega);

            consensus_ref_[problem][k](12) = consensus_dp[k][0] - dual_for_iteration[problem][k][0]/rho;
            consensus_ref_[problem][k](13) = consensus_dp[k][1] - dual_for_iteration[problem][k][1]/rho;
            consensus_ref_[problem][k](14) = consensus_dp[k][2] - dual_for_iteration[problem][k][2]/rho;
            consensus_ref_[problem][k](15) = consensus_omega[k][2] - dual_for_iteration[problem][k][5]/rho;
            consensus_ref_[problem][k](16) = consensus_omega[k][1] - dual_for_iteration[problem][k][4]/rho;
            consensus_ref_[problem][k](17) = consensus_omega[k][0] - dual_for_iteration[problem][k][3]/rho;

            // set dq
            counter=0;
            for(auto idx : config_param_.subsystems_map_joint[problem]) //循环6次
            {
                x_ref_[problem][k](18+counter) = reference.joint_velocity[k][idx];
                ++counter;
            }
            // set foot
            counter=0;
            for (auto idx : config_param_.subsystems_map_contact[problem]) //循环2*3次
            {
                x_ref_[problem][k](24+counter) = reference.foot_position[k][3*idx];
                x_ref_[problem][k](25+counter) = reference.foot_position[k][3*idx+1];
                x_ref_[problem][k](26+counter) = reference.foot_position[k][3*idx+2];
                counter+=3;
            }

            ////  ============ REFERENCE  U ============
            if (k < config_param_.N_) { // u N维
                // set tau
                counter=0;
                for(auto idx : config_param_.subsystems_map_joint[problem]) // 循环6次
                {
                    u_ref_[problem][k](counter) = reference.torque[k][idx];
                    ++counter;
                }

                // set grf //这里和原代码不同，我们只优化grf而不是grf_wb，因此只给当前子系统赋值即可
                counter=0;
                for(auto idx : config_param_.subsystems_map_contact[problem]) // 循环3*2=6次
                {
                    u_ref_[problem][k](6+counter) = reference.ground_reaction_force[k][3*idx];
                    u_ref_[problem][k](7+counter) = reference.ground_reaction_force[k][3*idx+1];
                    u_ref_[problem][k](8+counter) = reference.ground_reaction_force[k][3*idx+2];
                    counter+=3;              
                }

                // set grf aux
                counter=0;
                if (problem == "front") {
                    for(auto idx : config_param_.subsystems_map_contact["back"]) // 循环3*2=6次
                    {
                        u_ref_[problem][k](12+counter) = reference.ground_reaction_force[k][3*idx];
                        u_ref_[problem][k](13+counter) = reference.ground_reaction_force[k][3*idx+1];
                        u_ref_[problem][k](14+counter) = reference.ground_reaction_force[k][3*idx+2];
                        counter+=3;              
                    }
                } else {
                    for(auto idx : config_param_.subsystems_map_contact["front"]) // 循环3*2=6次
                    {
                        u_ref_[problem][k](12+counter) = reference.ground_reaction_force[k][3*idx];
                        u_ref_[problem][k](13+counter) = reference.ground_reaction_force[k][3*idx+1];
                        u_ref_[problem][k](14+counter) = reference.ground_reaction_force[k][3*idx+2];
                        counter+=3;              
                    }
                }
            }
        }

        ////  ============ WEIGHT  ============                
        // Contact-dependent weights are refreshed once per control cycle and
        // then kept constant over this QP's prediction horizon.
        {
            // weight p 
            Q.diagonal()[0] = weights.position[0];
            Q.diagonal()[1] = weights.position[1];
            Q.diagonal()[2] = weights.position[2];

            // weight quat
            Q.diagonal()[3] = weights.orientation[2];
            Q.diagonal()[4] = weights.orientation[1];
            Q.diagonal()[5] = weights.orientation[0];

            // weight q
            counter = 0;
            for(auto idx : config_param_.subsystems_map_joint[problem]) // 实际循环6次
            {
                Q.diagonal()[6+counter] = weights.joint_position;
                counter++;
            }
        
            // weight dp
            Q.diagonal()[12] = weights.linear_velocity[0];
            Q.diagonal()[13] = weights.linear_velocity[1];
            Q.diagonal()[14] = weights.linear_velocity[2];

            // weight omega
            Q.diagonal()[15] = weights.angular_velocity[2];
            Q.diagonal()[16] = weights.angular_velocity[1];
            Q.diagonal()[17] = weights.angular_velocity[0];

            Q_consensus.diagonal()[12] = weights.consensus;
            Q_consensus.diagonal()[13] = weights.consensus;
            Q_consensus.diagonal()[14] = weights.consensus;
            Q_consensus.diagonal()[15] = weights.consensus;
            Q_consensus.diagonal()[16] = weights.consensus;
            Q_consensus.diagonal()[17] = weights.consensus;

            // weight dq
            counter = 0;
            for(auto idx : config_param_.subsystems_map_joint[problem]) // 实际循环6次
            {
                Q.diagonal()[18+counter] = weights.joint_velocity;
                counter++;
            }

            // weight foot
            counter = 0;
            for(auto idx : config_param_.subsystems_map_contact[problem]) // 实际循环2*3次
            {   
                if (state.commanded_contact[idx] == 1)
                {
                    Q.diagonal()[24 + counter] = weights.foot_stance[0];
                    Q.diagonal()[25 + counter] = weights.foot_stance[1];
                    Q.diagonal()[26 + counter] = weights.foot_stance[2];
                }
                else
                {
                    Q.diagonal()[24 + counter] = weights.foot_swing[0];
                    Q.diagonal()[25 + counter] = weights.foot_swing[1];
                    Q.diagonal()[26 + counter] = weights.foot_swing[2];
                }
                counter+=3;
            }

            // weight tau
            counter = 0;
            for(auto idx : config_param_.subsystems_map_joint[problem]) // 实际循环6次
            {
                R.diagonal()[counter] = weights.torque;
                counter++;
            }

            // weight grf grf_aux
            counter = 0;
            for(auto idx : config_param_.subsystems_map_contact["wb"])
            {
                R.diagonal()[6+counter] = weights.ground_reaction_force;
                R.diagonal()[7+counter] = weights.ground_reaction_force;
                R.diagonal()[8+counter] = weights.ground_reaction_force;
                counter+=3;
            }

            // gamma
            gamma_[solver_index] = weights.gamma;
        }              
    }

    auto &front_x = x_candidate.at("front");
    auto &front_u = u_candidate.at("front");
    auto &back_x = x_candidate.at("back");
    auto &back_u = u_candidate.at("back");
    auto front = std::async(std::launch::async, [&] {
        return hpipmSolve(state, "front", 0, front_x, front_u);
    });
    auto back = std::async(std::launch::async, [&] {
        return hpipmSolve(state, "back", 1, back_x, back_u);
    });
    const bool front_solved = front.get();
    const bool back_solved = back.get();
    all_solved = front_solved && back_solved;

    if (!all_solved)
    {
        return false;
    }

    x_ = std::move(x_candidate);
    u_ = std::move(u_candidate);

    data_ = std::move(data_candidate);

    for (const auto &problem : config_param_.subsystems_name)
    {
        if (problem != "wb")
        {
            data_[problem].dual = std::move(dual_for_iteration[problem]);
        }
    }

    for (auto problem : config_param_.subsystems_name)
    {   
        if (problem == "wb")
            continue;
        // update state from solution

        std::vector<Eigen::VectorXd> &x = x_[problem];
        
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
            for(auto idx : config_param_.subsystems_map_joint[problem]) //6个循环
            {
                data_["wb"].q[k][idx] = x[k](6+counter);  // data_["wb"]放的是当前及预测状态，但是没放全   
                counter++;
            }
            
            //dp
            data_[problem].dp[k][0] = x[k](12);
            data_[problem].dp[k][1] = x[k](13);
            data_[problem].dp[k][2] = x[k](14);

            //omega
            data_[problem].omega[k][0] = x[k](17);
            data_[problem].omega[k][1] = x[k](16);
            data_[problem].omega[k][2] = x[k](15);

            //dq
            counter = 0;
            for(auto idx : config_param_.subsystems_map_joint[problem])  //6个循环
            {
                data_["wb"].dq[k][idx] = x[k](18+counter);
                counter++;
            }

            //foot
            counter = 0;
            for (auto idx : config_param_.subsystems_map_contact[problem]) //循环2*3次
            {
                data_["wb"].foot[k][3*idx]   = x[k](24+counter);
                data_["wb"].foot[k][3*idx+1] = x[k](25+counter);
                data_["wb"].foot[k][3*idx+2] = x[k](26+counter);
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
    // update whole body speeds
    for(int k{0};k<config_param_.N_+1;k++)
    {
        data_["wb"].dp[k][0] = 0;
        data_["wb"].dp[k][1] = 0;
        data_["wb"].dp[k][2] = 0;

        data_["wb"].omega[k][0] = 0;
        data_["wb"].omega[k][1] = 0;
        data_["wb"].omega[k][2] = 0;

        double n{static_cast<double>(config_param_.subsystems_name.size() - 1)}; //n=2

        for(auto problem : config_param_.subsystems_name)
        {
            if (problem == "wb")
                continue;
            data_["wb"].dp[k][0] += (data_[problem].dp[k][0] + data_[problem].dual[k][0]/rho)/n;
            data_["wb"].dp[k][1] += (data_[problem].dp[k][1] + data_[problem].dual[k][1]/rho)/n;
            data_["wb"].dp[k][2] += (data_[problem].dp[k][2] + data_[problem].dual[k][2]/rho)/n;

            data_["wb"].omega[k][0] += (data_[problem].omega[k][0] + data_[problem].dual[k][3]/rho)/n;
            data_["wb"].omega[k][1] += (data_[problem].omega[k][1] + data_[problem].dual[k][4]/rho)/n;
            data_["wb"].omega[k][2] += (data_[problem].omega[k][2] + data_[problem].dual[k][5]/rho)/n;
        }
    }
    // update dual
    for(auto problem : config_param_.subsystems_name)
    {
        if (problem == "wb")
        {
            continue;
        }
        for (int k{0};k<config_param_.N_+1;k++)
        {
            data_[problem].dual[k][0] += rho*(data_[problem].dp[k][0] - data_["wb"].dp[k][0]);
            data_[problem].dual[k][1] += rho*(data_[problem].dp[k][1] - data_["wb"].dp[k][1]);
            data_[problem].dual[k][2] += rho*(data_[problem].dp[k][2] - data_["wb"].dp[k][2]);
            data_[problem].dual[k][3] += rho*(data_[problem].omega[k][0] - data_["wb"].omega[k][0]);
            data_[problem].dual[k][4] += rho*(data_[problem].omega[k][1] - data_["wb"].omega[k][1]);
            data_[problem].dual[k][5] += rho*(data_[problem].omega[k][2] - data_["wb"].omega[k][2]);

            data_[problem].residual[k][0] = (data_[problem].dp[k][0] - data_["wb"].dp[k][0]);
            data_[problem].residual[k][1] = (data_[problem].dp[k][1] - data_["wb"].dp[k][1]);
            data_[problem].residual[k][2] = (data_[problem].dp[k][2] - data_["wb"].dp[k][2]);
            data_[problem].residual[k][3] = (data_[problem].omega[k][0] - data_["wb"].omega[k][0]);
            data_[problem].residual[k][4] = (data_[problem].omega[k][1] - data_["wb"].omega[k][1]);
            data_[problem].residual[k][5] = (data_[problem].omega[k][2] - data_["wb"].omega[k][2]);
        }
        // for (int k{0};k<config_param_.N_;k++)
        // {
        //     data_[problem].dual[k][0] = (data_["front"].dp[k][0] - (data_["back"].dp[k][0]))*weights.consensus;
        //     data_[problem].dual[k][1] = (data_["front"].dp[k][1] - (data_["back"].dp[k][1]))*weights.consensus;
        //     data_[problem].dual[k][2] = (data_["front"].dp[k][2] - (data_["back"].dp[k][2]))*weights.consensus;
        //     data_[problem].dual[k][3] = (data_["front"].omega[k][0] - (data_["back"].omega[k][0]))*weights.consensus;
        //     data_[problem].dual[k][4] = (data_["front"].omega[k][1] - (data_["back"].omega[k][1]))*weights.consensus;
        //     data_[problem].dual[k][5] = (data_["front"].omega[k][2] - (data_["back"].omega[k][2]))*weights.consensus;
        // }
    }
    // check stopping criteria

    do_init = false;
    return true;
}

MpcResult codmpcSolver::getResult(const ContactVector &contact, bool success) const
{
    MpcResult result;
    result.success = success;
    result.contact = contact;
    if (data_.empty() || data_.at("wb").q.empty())
    {
        return result;
    }

    result.joint_position = Eigen::Map<const JointVector>(data_.at("wb").q[1].data());
    result.joint_velocity = Eigen::Map<const JointVector>(data_.at("wb").dq[1].data());
    result.torque = Eigen::Map<const JointVector>(data_.at("wb").tau[0].data());

    result.snapshot.position = Eigen::Map<const Eigen::Vector3d>(data_.at("front").p[0].data());
    result.snapshot.rpy = Eigen::Map<const Eigen::Vector3d>(data_.at("front").rpy[0].data());
    result.snapshot.linear_velocity = Eigen::Map<const Eigen::Vector3d>(data_.at("front").dp[0].data());
    result.snapshot.rpy_rate = Eigen::Map<const Eigen::Vector3d>(data_.at("front").omega[0].data());
    result.snapshot.foot_position = Eigen::Map<const FootVector>(data_.at("wb").foot[0].data());
    result.snapshot.joint_position = Eigen::Map<const JointVector>(data_.at("wb").q[0].data());
    result.snapshot.joint_velocity = Eigen::Map<const JointVector>(data_.at("wb").dq[0].data());
    result.snapshot.torque = result.torque;
    result.snapshot.ground_reaction_force = Eigen::Map<const FootVector>(data_.at("wb").grf[0].data());
    return result;
}

bool codmpcSolver::hpipmSolve(const RobotState &state,
                              const std::string &subsystems_name,
                              std::size_t solver_index,
                              std::vector<Eigen::VectorXd> &x_candidate,
                              std::vector<Eigen::VectorXd> &u_candidate) {
    // setup QP
    int s_idx = 0;
    if (subsystems_name == "front") {
        s_idx = 0;
    } else if (subsystems_name == "back") {
        s_idx = 2;
    } else {
        return false;
    }

    int const &N = config_param_.N_;
    int const &nx = config_param_.n_state;
    int const &nu = config_param_.n_control;
    int const &n_contact_wb = config_param_.n_contact_wb;
    const Eigen::VectorXd &x0 = x0_.at(subsystems_name);
    const std::vector<Eigen::VectorXd> &x_ref = x_ref_.at(subsystems_name);
    const std::vector<Eigen::VectorXd> &consensus_ref = consensus_ref_.at(subsystems_name);
    const std::vector<Eigen::VectorXd> &u_ref = u_ref_.at(subsystems_name);

    auto &qp = qp_[solver_index];

    // dynamics
    const Eigen::MatrixXd &A = quadruped_model_.Ak_.at(subsystems_name);
    const Eigen::MatrixXd &B = quadruped_model_.Bk_.at(subsystems_name);
    const Eigen::VectorXd &b = quadruped_model_.bk_.at(subsystems_name);
    for (int i=0; i<N; ++i) { //0～N-1
        qp[i].A = A;
        qp[i].B = B;
        qp[i].b = b;
    }

    // cost
    Eigen::MatrixXd Q = Q_[solver_index];
    Eigen::MatrixXd Q_consensus = Q_consensus_[solver_index];
    Eigen::MatrixXd Q_total = Q + Q_consensus;
    Eigen::MatrixXd R = R_[solver_index];
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(nu, nx);

    // const Eigen::VectorXd q = - Q * x_ref;
    // const Eigen::VectorXd r = Eigen::VectorXd::Zero(nu);
    Eigen::VectorXd q = Eigen::VectorXd::Zero(nx);
    Eigen::VectorXd r = Eigen::VectorXd::Zero(nu);
    for (int i=0; i<N; ++i) { //0～N-1
        q = - Q * x_ref[i] - Q_consensus * consensus_ref[i];
        r = - R * u_ref[i];
        qp[i].Q = Q_total;
        qp[i].R = R;
        qp[i].S = S;
        qp[i].q = q;
        qp[i].r = r;
        Q_total *= gamma_[solver_index];
        Q *= gamma_[solver_index];
        Q_consensus *= gamma_[solver_index];
        // R *= gamma_;
    }
    q = - Q * x_ref[N] - Q_consensus * consensus_ref[N];
    qp[N].Q = Q_total;
    qp[N].q = q;

    // The torque bounds are frozen with the QP and applied at every control stage.
    const auto &joint_indices = config_param_.subsystems_map_joint.at(subsystems_name);
    const int n_joints = static_cast<int>(joint_indices.size());
    std::vector<int> torque_indices(n_joints);
    Eigen::VectorXd torque_min(n_joints);
    Eigen::VectorXd torque_max(n_joints);
    for (int i{0}; i < n_joints; ++i)
    {
        torque_indices[i] = i;
        const double limit = config_param_.torque_limit.at(joint_indices[i]);
        torque_min[i] = -limit;
        torque_max[i] = limit;
    }
    for (int i{0}; i < N; ++i)
    {
        qp[i].idxbu = torque_indices;
        qp[i].lbu = torque_min;
        qp[i].ubu = torque_max;
    }

    // constraints
    /////////////////// constrain 1: foot noslip
    double const epsilon = config_param_.no_slip_velocity;
    int n_noslip_constrain = 2*3;
    const auto &contact_cmd = state.commanded_contact;
    Eigen::MatrixXd J_matrix = Eigen::MatrixXd::Zero(n_noslip_constrain, 12);
    J_matrix.block(0, 0, 3, 12) = contact_cmd[s_idx]*quadruped_model_.J_linear_sub_[s_idx];
    J_matrix.block(3, 0, 3, 12) = contact_cmd[s_idx+1]*quadruped_model_.J_linear_sub_[s_idx+1];
    // J_matrix.block(0, 0, 3, 12) = contact_cmd[s_idx]*quadruped_model_.local_J_linear_sub_[s_idx];
    // J_matrix.block(3, 0, 3, 12) = contact_cmd[s_idx+1]*quadruped_model_.local_J_linear_sub_[s_idx+1];
    Eigen::MatrixXd J_select = Eigen::MatrixXd::Zero(n_noslip_constrain, nx);
    J_select.block(0, 12, n_noslip_constrain, 12) = J_matrix;
    Eigen::VectorXd vec_foot_vel_max = epsilon*Eigen::VectorXd::Ones(n_noslip_constrain);
    Eigen::VectorXd vec_foot_vel_min = -vec_foot_vel_max;

    //////////////////// constrain 2: friction cone
    double const mu = config_param_.friction_coefficient;
    double const fz_max = config_param_.normal_force_max;
    double const fz_min = config_param_.normal_force_min;
    
    int n_friction_cone_constrain = 4*5;
    Eigen::MatrixXd friction_matrix_block(5, 3);
    friction_matrix_block << 1,  0, mu,
                            -1,  0, mu,
                             0,  1, mu,
                             0, -1, mu,
                             0,  0, 1;

    Eigen::MatrixXd friction_matrix = Eigen::MatrixXd::Zero(n_friction_cone_constrain, nu);
    for (int i=0; i<n_contact_wb; ++i) {
        friction_matrix.block(5*i, 6+3*i, 5, 3) = friction_matrix_block;
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

    const int num_constraints = n_noslip_constrain + n_friction_cone_constrain;
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(num_constraints, nx);
    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(num_constraints, nu);
    Eigen::VectorXd lg = Eigen::VectorXd::Zero(num_constraints);
    Eigen::VectorXd ug = Eigen::VectorXd::Zero(num_constraints);

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
    auto &solution = solution_[solver_index];
    const auto status = hpipm_solver_[solver_index].solve(x0, qp, solution);

    if (status == hpipm::HpipmStatus::Success) {
        x_candidate.resize(N+1);
        u_candidate.resize(N);
        for (int i = 0; i < N; ++i) {
            u_candidate[i] = solution[i].u;
            x_candidate[i] = solution[i].x;
        }
        x_candidate[N] = solution[N].x;

        return true;
    } else {
        std::cerr << "HPIPM solving failed! Error code: " << status << std::endl;
    }

    return false;
}
