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

#ifdef USE_QPOASES
#include "qpOASES.hpp"
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
        void init(const parameter &solver_param);
        void solve( bool &do_init,
                    const std::map<std::string,std::vector<double>> &initial_condition,
                    const std::map<std::string,std::vector<std::vector<double>>> &ref,
                    const std::map<std::string,std::vector<std::vector<double>>> &param,
                    const std::map<std::string,std::vector<double>> &weight_vec);
        void getControl(std::vector<double> &des_q,std::vector<double> &des_dq,std::vector<double> &des_tau);
        void getData(std::map<std::string,pdata> &data);
        void prepare(); 
        void sendSolverData(std::vector<std::vector<double>> const &reference, std::vector<double> const &initial_condition, std::vector<double> const &u0_init);
        void receiveSolverResult();
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> Q_;
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> R_;

#ifdef USE_QPOASES
        void computeQPmatrices(std::string const &subsystems_name, 
                      std::vector<double> const &problem_initial_condition, 
                      std::vector<std::vector<double>> const &problem_ref,
                      std::vector<std::vector<double>> const &problem_ref_u,
                      Eigen::MatrixXd& H, Eigen::VectorXd& g);
        bool qpOASESsolve(std::vector<double> const &problem_initial_condition, 
                                std::vector<std::vector<double>> const &problem_ref,
                                std::vector<std::vector<double>> const &problem_ref_u,
                                std::string const &subsystems_name);
#endif
    
    private:
        parameter solver_param_;
        quadrupedModel quadruped_model_;
        std::map<std::string, pdata> data_;
        std::map<std::string, std::vector<std::vector<double>>> u_;
        std::map<std::string, std::vector<double>> x0_;
};

#endif
