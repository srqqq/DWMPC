#ifndef CODMPC_SOLVER_HPP
#define CODMPC_SOLVER_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <map>
#include <string>
#include <stdexcept>
#include <array>
#include "controllers/dwmpc/pinocchio_model.hpp"
#include "controllers/dwmpc/types.hpp"

#include "hpipm-cpp/hpipm-cpp.hpp"

class pdata
{   
    public:
    std::vector<std::vector<double>> p{}; // position
    std::vector<std::vector<double>> quat{}; // quaternion
    std::vector<std::vector<double>> rpy{}; // roll pitch yaw
    std::vector<std::vector<double>> q{}; // joint angle
    std::vector<std::vector<double>> dp{}; // linear velocity prediction
    std::vector<std::vector<double>> omega{}; // roll-pitch-yaw rates
    std::vector<std::vector<double>> dq{}; // joint velocity
    std::vector<std::vector<double>> grf{}; // ground reaction forces
    std::vector<std::vector<double>> tau{}; // joint torque
    std::vector<std::vector<double>> foot{}; // foot position
    std::vector<std::vector<double>> dual{}; //dual 
    std::vector<std::vector<double>> residual{}; //residual
};

class codmpcSolver {
    public:
        codmpcSolver();
        virtual ~codmpcSolver();
        void init(const parameter &config_param);
        bool solve(bool &do_init,
                   const RobotState &state,
                   const ReferenceTrajectory &reference,
                   const MpcWeights &weights);
        MpcResult getResult(const ContactVector &contact, bool success) const;
        const std::map<std::string,pdata> &getData() const { return data_; }

    private:
        bool hpipmSolve(const RobotState &state,
                        const std::string &subsystems_name,
                        std::size_t solver_index,
                        std::vector<Eigen::VectorXd> &x_candidate,
                        std::vector<Eigen::VectorXd> &u_candidate);
        parameter config_param_;
        quadrupedModel quadruped_model_;
        std::map<std::string, pdata> data_;
        std::map<std::string, std::vector<Eigen::VectorXd>> u_; // 优化结果控制序列
        std::map<std::string, std::vector<Eigen::VectorXd>> x_; // 优化结果状态序列
        std::map<std::string, Eigen::VectorXd> x0_; //初始状态
        std::map<std::string, std::vector<Eigen::VectorXd>> x_ref_; // 参考状态序列
        std::map<std::string, std::vector<Eigen::VectorXd>> u_ref_; // 参考输入序列
        std::map<std::string, std::vector<Eigen::VectorXd>> consensus_ref_; // 参考一致项序列
        std::array<Eigen::DiagonalMatrix<double, Eigen::Dynamic>, 2> Q_;
        std::array<Eigen::DiagonalMatrix<double, Eigen::Dynamic>, 2> Q_consensus_;
        std::array<Eigen::DiagonalMatrix<double, Eigen::Dynamic>, 2> R_;
        std::array<double, 2> gamma_{};
        std::array<std::vector<hpipm::OcpQp>, 2> qp_;
        std::array<std::vector<hpipm::OcpQpSolution>, 2> solution_;
        std::array<hpipm::OcpQpIpmSolver, 2> hpipm_solver_;
};

#endif
