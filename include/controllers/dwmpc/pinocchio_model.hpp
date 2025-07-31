#ifndef PINOCCHIO_MODEL_HPP
#define PINOCCHIO_MODEL_HPP

#include <iostream>
#include "yaml-cpp/yaml.h"
#include <Eigen/Dense>
#include <numeric>
#include "controllers/dwmpc/rotation.hpp"
#include "pinocchio/fwd.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/compute-all-terms.hpp"
#include "pinocchio/parsers/urdf.hpp"

class parameter
{   public:
    int max_iteration{100}; // maximum number of iteration for the distributed solver
    bool receding_horizon{true}; // flag if using a reciding horizon
    std::vector<std::string> subsystems_name; //vecotr of the names of the subsystems
    std::map<std::string,std::vector<int>> subsystems_map_joint;//nap the name of the subsistem to the joint number in the whole body 
    std::map<std::string,std::vector<int>> subsystems_map_contact;//nap the name of the subsistem to the contact number in the whole body 
    int n_problem{}; // number of the subsystems
    int n_contact_wb{};
    int n_contact{};
    int n_joint_wb{};
    int n_joint{};
    int N_{};// lenght of the horizon  
    int n_state{};
    int n_control{};
};

class quadrupedModel {

    public:
    quadrupedModel();
    ~quadrupedModel();
    void modelInit(parameter const &model_param);
    void modelUpdate(std::map<std::string,std::vector<double>> const &xk);
    std::vector<std::vector<double>> updatePrediction(std::vector<double> const &x0,
                                                      std::vector<std::vector<double>> const &u,
                                                      std::string const &subsystems_name);
    std::map<std::string, Eigen::MatrixXd> Ak_;
    std::map<std::string, Eigen::MatrixXd> Bk_;

    private:
    void updateSubsystem(std::string &subsystems_name, Eigen::MatrixXd const &M_wb, 
                         Eigen::VectorXd const &nle_wb, Eigen::MatrixXd const &inv_jac_R,
                         std::map<std::string,std::vector<double>> const &xk);
    pinocchio::Model pin_model_;
    pinocchio::Data pin_data_;
    parameter model_param_;
    std::vector<std::string> subsystems_name_list_;
    std::vector<std::string> contact_frame_name_list_wb_;
    std::vector<Eigen::MatrixXd> J_linear_; //足端线速度雅可比矩阵
};

double normalizeAngle(double angle);

#endif
