#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "controllers/dwmpc/dwmpc.hpp"// Include the header file for the Dwmpc class

namespace py = pybind11;
Eigen::Quaterniond numpy_to_quaternion(const py::array_t<double>& array) {
    auto buf = array.unchecked<1>();  // Unpack the NumPy array
    return Eigen::Quaterniond(buf(0), buf(1), buf(2), buf(3));  // w, x, y, z order
}
namespace controllers
{
PYBIND11_MODULE(pydwmpc, m) {
    py::class_<pdata>(m, "pdata")
        .def(py::init<>())
        .def_readwrite("p", &pdata::p, "Position")
        .def_readwrite("quat", &pdata::quat, "Quaternion")
        .def_readwrite("rpy", &pdata::rpy, "Roll Pitch Yaw")
        .def_readwrite("q", &pdata::q, "Joint angles")
        .def_readwrite("dp", &pdata::dp, "Linear velocity prediction")
        .def_readwrite("omega", &pdata::omega, "Angular velocity")
        .def_readwrite("dq", &pdata::dq, "Joint velocity")
        .def_readwrite("grf", &pdata::grf, "Ground reaction forces")
        .def_readwrite("tau", &pdata::tau, "Joint torque")
        .def_readwrite("foot", &pdata::foot, "Foot Position")
        .def_readwrite("dual", &pdata::dual, "Dual variables")
        .def_readwrite("residual", &pdata::residual, "Residuals");
    py::class_<MpcSnapshot>(m, "MpcSnapshot")
        .def_readonly("position", &MpcSnapshot::position)
        .def_readonly("rpy", &MpcSnapshot::rpy)
        .def_readonly("linear_velocity", &MpcSnapshot::linear_velocity)
        .def_readonly("rpy_rate", &MpcSnapshot::rpy_rate)
        .def_readonly("foot_position", &MpcSnapshot::foot_position)
        .def_readonly("joint_position", &MpcSnapshot::joint_position)
        .def_readonly("joint_velocity", &MpcSnapshot::joint_velocity)
        .def_readonly("torque", &MpcSnapshot::torque)
        .def_readonly("ground_reaction_force", &MpcSnapshot::ground_reaction_force);
    py::class_<MpcResult>(m, "MpcResult")
        .def_readonly("success", &MpcResult::success)
        .def_readonly("contact", &MpcResult::contact)
        .def_readonly("torque", &MpcResult::torque)
        .def_readonly("joint_position", &MpcResult::joint_position)
        .def_readonly("joint_velocity", &MpcResult::joint_velocity)
        .def_readonly("snapshot", &MpcResult::snapshot);
    py::class_<Dwmpc>(m, "Dwmpc")
        .def(py::init<>())
        .def("init", &Dwmpc::init)
        .def("run", &Dwmpc::run,
             py::arg("p"), py::arg("quat"), py::arg("q_op"), py::arg("dp"), py::arg("omega"), py::arg("dq_op"),
             py::arg("loop_dt"), py::arg("current_contact"), py::arg("foot_op"), py::arg("desired_linear_speed"),
             py::arg("desired_angular_speed"),
             "All floating-base velocities, foot positions, and desired base velocities use the world frame.")
        .def("setWeight", &Dwmpc::setWeight)
        .def("setGaitParam", py::overload_cast<const double, const double, const std::vector<double>&>(&Dwmpc::setGaitParam))
        .def("setGaitParam", py::overload_cast<const double, const double, const int>(&Dwmpc::setGaitParam))
        .def("updateTimer", &Dwmpc::updateTimer)
        .def("reset", &Dwmpc::reset)
        .def("startWalking", &Dwmpc::startWalking)
        .def("stopWalking", &Dwmpc::stopWalking)
        .def("setSineParam", &Dwmpc::setSineParam)
        .def("startSineWave", &Dwmpc::startSineWave)
        .def("stopSineWave", &Dwmpc::stopSineWave)
        .def("getWeight", &Dwmpc::getWeight)
        .def("setStepHeight", &Dwmpc::setStepHeight)
        .def("getFullPrediction", &Dwmpc::getFullPrediction,
             py::return_value_policy::reference_internal,
             "Retrieve the full prediction data");
}
} //namespace controllers
