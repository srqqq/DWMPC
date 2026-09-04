#ifndef DWMPC_TYPES_HPP
#define DWMPC_TYPES_HPP

#include <Eigen/Dense>
#include <vector>

constexpr int kNumJoints = 12;
constexpr int kNumContacts = 4;
constexpr int kStateDim = 30;
constexpr int kControlDim = 18;

using JointVector = Eigen::Matrix<double, kNumJoints, 1>;
using ContactVector = Eigen::Matrix<double, kNumContacts, 1>;
using FootVector = Eigen::Matrix<double, 3*kNumContacts, 1>;

struct RobotState
{
    // Floating-base and foot Cartesian quantities use the world frame.
    Eigen::Vector3d position;
    Eigen::Vector3d rpy;
    JointVector joint_position;
    Eigen::Vector3d linear_velocity;
    Eigen::Vector3d angular_velocity;
    JointVector joint_velocity;
    ContactVector measured_contact;
    ContactVector commanded_contact;
    FootVector foot_position;
};

struct ReferenceTrajectory
{
    // Base, foot and ground-reaction-force Cartesian quantities use the world frame.
    std::vector<Eigen::Vector3d> position;
    std::vector<Eigen::Vector3d> rpy;
    std::vector<JointVector> joint_position;
    std::vector<Eigen::Vector3d> linear_velocity;
    std::vector<Eigen::Vector3d> angular_velocity;
    std::vector<JointVector> joint_velocity;
    std::vector<JointVector> torque;
    std::vector<FootVector> ground_reaction_force;
    std::vector<FootVector> foot_position;
    std::vector<ContactVector> contact_schedule;
};

struct MpcWeights
{
    Eigen::Vector3d position;
    Eigen::Vector3d orientation;
    double joint_position;
    Eigen::Vector3d linear_velocity;
    Eigen::Vector3d angular_velocity;
    double joint_velocity;
    double torque;
    double ground_reaction_force;
    Eigen::Vector3d foot_stance;
    Eigen::Vector3d foot_swing;
    double consensus;
    double gamma;
};

struct MpcSnapshot
{
    Eigen::Vector3d position;
    Eigen::Vector3d rpy;
    Eigen::Vector3d linear_velocity;
    Eigen::Vector3d rpy_rate;
    FootVector foot_position;
    JointVector joint_position;
    JointVector joint_velocity;
    JointVector torque;
    FootVector ground_reaction_force;
};

struct MpcResult
{
    bool success{false};
    ContactVector contact{ContactVector::Zero()};
    JointVector torque{JointVector::Zero()};
    JointVector joint_position{JointVector::Zero()};
    JointVector joint_velocity{JointVector::Zero()};
    MpcSnapshot snapshot{};
};

#endif
