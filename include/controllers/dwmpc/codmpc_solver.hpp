#ifndef CODMPC_SOLVER_HPP
#define CODMPC_SOLVER_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <map>
#include <cstring>
#include <cstdio>
#include <stdlib.h>
#include <csignal>
#include <string>
#include "controllers/dwmpc/pinocchio_model.hpp"
#ifdef DEBUG_MODE
#include "controllers/dwmpc/robot_data_logger.hpp"
#endif

#ifdef USE_QPOASES
#include "qpOASES.hpp"
#endif

#ifdef USE_HPIPM
#include "hpipm-cpp/hpipm-cpp.hpp"
#endif

class pdata
{   
    public:
    std::vector<std::vector<double>> p{}; // position
    std::vector<std::vector<double>> quat{}; // quaternion
    std::vector<std::vector<double>> rpy{}; // roll pitch yaw
    std::vector<std::vector<double>> q{}; // joint angle
    std::vector<std::vector<double>> dp{}; // linear velocity prediction
    std::vector<std::vector<double>> omega{}; //angular velocity 
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
        void init(const parameter &config_param);
        void solve( bool &do_init,
                    const std::map<std::string,std::vector<double>> &x0_map,
                    const std::map<std::string,std::vector<std::vector<double>>> &ref,
                    const std::map<std::string,std::vector<std::vector<double>>> &param,
                    const std::map<std::string,std::vector<double>> &weight_vec);
        void getControl(std::vector<double> &des_q,std::vector<double> &des_dq,std::vector<double> &des_tau);
        void getData(std::map<std::string,pdata> &data);
        void prepare(); 
        // void sendSolverData(std::vector<std::vector<double>> const &reference, std::vector<double> const &initial_condition, std::vector<double> const &u0_init);
        // void receiveSolverResult();
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> Q_;
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> R_;
        double gamma_;

#ifdef USE_HPIPM
        bool hpipmSolve(Eigen::VectorXd const &x0, std::map<std::string,std::vector<double>> const &x0_map,
                        std::vector<Eigen::VectorXd> const &x_ref,
                        std::vector<Eigen::VectorXd> const &u_ref,
                        std::string const &subsystems_name);
#endif

#ifdef USE_QPOASES
        // 权重对角矩阵
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> Q_total_;
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> R_total_;
        Eigen::MatrixXd R_total_dense_;
        
        void buildTotalWeightMatrices();
        void buildFMatrix(Eigen::MatrixXd &F, Eigen::MatrixXd const &A);
        void buildPhiMatrix(Eigen::MatrixXd &Phi, Eigen::MatrixXd const &A, Eigen::MatrixXd const &B);
        void qpOASESinit();
        void computeQPmatrices(std::string const &subsystems_name,
                               Eigen::VectorXd const &x0, std::map<std::string,std::vector<double>> const &x0_map,
                               std::vector<Eigen::VectorXd> const &x_ref,
                               std::vector<Eigen::VectorXd> const &u_ref,
                               Eigen::MatrixXd& H, Eigen::VectorXd& g, 
                               Eigen::MatrixXd& Ac, Eigen::VectorXd& lbAc, Eigen::VectorXd& ubAc);
        bool qpOASESsolve(Eigen::VectorXd const &x0, std::map<std::string,std::vector<double>> const &x0_map,
                                std::vector<Eigen::VectorXd> const &x_ref,
                                std::vector<Eigen::VectorXd> const &u_ref,
                                std::string const &subsystems_name);
#endif
    private:
        parameter config_param_;
        quadrupedModel quadruped_model_;
        std::map<std::string, pdata> data_;
        std::map<std::string, std::vector<Eigen::VectorXd>> u_; // 优化结果控制序列
        std::map<std::string, std::vector<Eigen::VectorXd>> x_; // 优化结果状态序列
        std::map<std::string, Eigen::VectorXd> x0_;
        int constrains_;
        bool is_solver_initialized{false};

#ifdef DEBUG_MODE
        RobotDataLogger logger_front_;
        RobotDataLogger logger_back_;
#endif
};

#endif
