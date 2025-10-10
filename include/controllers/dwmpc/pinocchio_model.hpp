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
    void modelUpdate(std::map<std::string,std::vector<double>> const &x0_map);
    std::vector<Eigen::VectorXd> updatePrediction(Eigen::VectorXd const &x0,
                                                std::vector<Eigen::VectorXd> const &u,
                                                std::string const &subsystems_name);
    void createSelectMatrix(std::string const &subsystems_name, std::map<std::string, std::vector<double>> const &x0_map,
                            Eigen::MatrixXd &S);

    std::map<std::string, Eigen::MatrixXd> Ak_;
    std::map<std::string, Eigen::MatrixXd> Bk_;

    std::vector<Eigen::MatrixXd> J_linear_wb_; //足端线速度雅可比矩阵
    std::vector<Eigen::MatrixXd> J_linear_sub_; //子系统足端线速度雅可比矩阵
    std::vector<Eigen::MatrixXd> J_linear_leg_; //单腿足端线速度雅可比矩阵，维度3*3
    // std::vector<Eigen::MatrixXd> J_linear_submix_; //足端线速度雅可比矩阵,基座+单腿

    // std::vector<Eigen::MatrixXd> world_J_linear_wb_; //world系下的足端线速度雅可比矩阵
    // std::vector<Eigen::MatrixXd> world_J_linear_sub_; //world系下的子系统足端线速度雅可比矩阵
    // std::vector<Eigen::MatrixXd> world_J_linear_leg_; //world系下单腿足端线速度雅可比矩阵，维度3*3
    // std::vector<Eigen::MatrixXd> world_J_linear_submix_; //world系下的子系统足端线速度雅可比矩阵,基座+单腿

    private:
    void updateSubsystem(std::string const &subsystems_name, Eigen::MatrixXd const &M_wb, 
                         Eigen::VectorXd const &nle_wb, Eigen::MatrixXd const &inv_jac_R,
                         std::map<std::string,std::vector<double>> const &x0_map);
    pinocchio::Model pin_model_;
    pinocchio::Data pin_data_;
    parameter model_param_;
    std::vector<std::string> subsystems_name_list_;
    std::vector<std::string> contact_frame_name_list_wb_;
};

double normalizeAngle(double angle);

#ifdef DEBUG_MODE

void debug_print(const std::vector<double>& vec);
void debug_print(const std::vector<std::vector<double>>& mat);
void debug_print(const Eigen::VectorXd& vec);
void debug_print(const Eigen::MatrixXd& mat);

#endif

#endif
