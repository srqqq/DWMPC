#ifndef DWMPC_HPP
#define DWMPC_HPP

#include <iostream>
#include <stdexcept>
#include "yaml-cpp/yaml.h"
#include "controllers/dwmpc/timer.hpp"

#include "ndcurves/bezier_curve.h"
#include <Eigen/Dense>
#include "controllers/dwmpc/codmpc_solver.hpp"
#include <numeric>

#include "controllers/dwmpc/rotation.hpp"

typedef double timer_param_t;
typedef Eigen::VectorXd pointX_t;
typedef double num_t;
typedef ndcurves::bezier_curve <timer_param_t, num_t, true, pointX_t> bezier_curves_t; 
// add other includes here

namespace controllers
{    
    class Dwmpc
    {
        public:
        Dwmpc();
        ~Dwmpc();
        // Floating-base position/orientation, base velocities, foot positions,
        // and desired base velocities are all expressed in the world frame.
        // Joint states and contact flags are frame-independent interface data.
        MpcResult run(const Eigen::Ref<const Eigen::VectorXd> &p,
                 const Eigen::Ref<Eigen::Vector4d> &quat,
                 const Eigen::Ref<const Eigen::VectorXd> &q_op,
                 const Eigen::Ref<const Eigen::VectorXd> &dp,
                 const Eigen::Ref<const Eigen::VectorXd> &omega,
                 const Eigen::Ref<const Eigen::VectorXd> &dq_op,
                 const double &loop_dt,
                 const Eigen::Ref<const Eigen::Vector4d> &current_contact,
                 const Eigen::Ref<const Eigen::MatrixXd> &foot_op,
                 const Eigen::Ref<const Eigen::VectorXd> &desired_linear_speed,
                 const Eigen::Ref<const Eigen::VectorXd> &desired_angular_speed);
        void init();
        void setWeight(const std::map<std::string,std::vector<double>> &weight_vec);
        void setGaitParam(const double duty_factor, const double step_freq, const std::vector<double> &delta);
        void setGaitParam(const double duty_factor, const double step_freq, const int gait_type);
        void updateTimer(const std::vector<double> &t, const std::vector<bool> &init);
        void reset();
        void startWalking();
        void stopWalking();
        const std::map<std::string,pdata> &getFullPrediction() const;
        void setSineParam(double frequency, double amplitude);
        void startSineWave();
        void stopSineWave();
        std::map<std::string,std::vector<double>> getWeight(){return weight_vec_;};
        void setStepHeight(double step_height);
    private:
        YAML::Node config;
        double proprioHeight(double desired_height);
        void setDesiredAndParameter(const ContactVector &contact0,
                                    const Eigen::MatrixXd &foot_op,
                                    const RobotState &state,
                                    ReferenceTrajectory &reference);
        void sineWave(ReferenceTrajectory &reference);
        void updateTerrainHeight(const ContactVector &contact0,
                                 const Eigen::MatrixXd &foot_op);
        codmpcSolver ocp_;
        int N_;
        double dt_;

        int n_joint_wb_;
        int n_contact_wb_;
        std::vector<bool> early_contact_;
        double time_{0};
        double frequency_{1};
        double amplitude_{0.1};
        bool do_sine_wave_{false};
        bool do_init_{true};
        
        std::vector<double> terrain_height_; // terrain height for each leg
        std::map<std::string,std::vector<double>> weight_vec_; // cost function weights
        MpcWeights weights_;
        Timer timer_; //timer to define the gait
        std::vector<bezier_curves_t> bcs_{4}; // TODO CHANGE THIS
        std::vector<Eigen::Vector3d> liftoff_pos_{4}; // liftoff position for each leg
        std::map<std::string,std::vector<double>> desired_; // desired values for the controller

        // TO DO change this to separate class that reads from a config file
        std::vector<double> q0_ {}; // home joint angle
        std::vector<double> foot0_ {}; // home foot position 
    };
} //namespace controllers

#endif /* end of include guard: DWMPC_HPP */
