#include "controllerbase.h"
#include "yaml-cpp/yaml.h"
#include <chrono>
#include <thread>
#include <unistd.h>
#include <stdio.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <iostream>
#include <functional>
namespace py = pybind11;
using namespace legged;
using Pose6d = Eigen::Matrix<double, 6, 1>;
using VecDoF = Eigen::VectorXd;


PYBIND11_MODULE(noetix_interface, m)
{
    // m.def("setcallback", &setcallback, "Set the callback function");
    // m.def("trigger_callback", &trigger_callback, "Trigger the registered callback");
    py::class_<JointState>(m, "JointState")
        .def(py::init<>())
        .def_readwrite("arm_l1_joint", &JointState::arm_l1_joint)
        .def_readwrite("arm_l2_joint", &JointState::arm_l2_joint)
        .def_readwrite("arm_l3_joint", &JointState::arm_l3_joint)
        .def_readwrite("arm_l4_joint", &JointState::arm_l4_joint)
        .def_readwrite("leg_l1_joint", &JointState::leg_l1_joint)
        .def_readwrite("leg_l2_joint", &JointState::leg_l2_joint)
        .def_readwrite("leg_l3_joint", &JointState::leg_l3_joint)
        .def_readwrite("leg_l4_joint", &JointState::leg_l4_joint)
        .def_readwrite("leg_l5_joint", &JointState::leg_l5_joint)
        .def_readwrite("arm_r1_joint", &JointState::arm_r1_joint)
        .def_readwrite("arm_r2_joint", &JointState::arm_r2_joint)
        .def_readwrite("arm_r3_joint", &JointState::arm_r3_joint)
        .def_readwrite("arm_r4_joint", &JointState::arm_r4_joint)
        .def_readwrite("leg_r1_joint", &JointState::leg_r1_joint)
        .def_readwrite("leg_r2_joint", &JointState::leg_r2_joint)
        .def_readwrite("leg_r3_joint", &JointState::leg_r3_joint)
        .def_readwrite("leg_r4_joint", &JointState::leg_r4_joint)
        .def_readwrite("leg_r5_joint", &JointState::leg_r5_joint);
    py::class_<MotorState>(m, "MotorState")
        .def(py::init<>())
        .def_readwrite("pos", &MotorState::pos)
        .def_readwrite("vel", &MotorState::vel)
        .def_readwrite("tau", &MotorState::tau)
        .def_readwrite("motor_id", &MotorState::motor_id)
        .def_readwrite("error", &MotorState::error);
    py::class_<MotorCmd>(m, "MotorCmd")
        .def(py::init<>())
        .def_readwrite("pos", &MotorCmd::pos)
        .def_readwrite("vel", &MotorCmd::vel)
        .def_readwrite("tau", &MotorCmd::tau)
        .def_readwrite("kp", &MotorCmd::kp)
        .def_readwrite("kd", &MotorCmd::kd)
        .def_readwrite("motor_id", &MotorCmd::motor_id);
    py::class_<NingImuData>(m, "NingImuData")
        .def(py::init<>())
        .def_property("ori",[](const NingImuData& s) {return py::array_t<double>({4}, {sizeof(double)}, s.ori);},nullptr)
        .def_property("ori_cov",[](const NingImuData& s) {return py::array_t<double>({9},{sizeof(double)},s.ori_cov);},nullptr)
        .def_property("angular_vel", [](const NingImuData& s) {return py::array_t<double>({3},{sizeof(double)},s.angular_vel);},nullptr)
        .def_property("angular_vel_cov",[](const NingImuData& s) {return py::array_t<double>({9},{sizeof(double)},s.angular_vel_cov);},nullptr)
        .def_property("linear_acc", [](const NingImuData& s) {return py::array_t<double>({3},{sizeof(double)},s.linear_acc);},nullptr)
        .def_property("linear_acc_cov", [](const NingImuData& s) {return py::array_t<double>({9},{sizeof(double)},s.linear_acc_cov);},nullptr);
    py::class_<joydata>(m, "joydata")
        .def(py::init<>())
        .def_property_readonly("axes",[](const joydata& j) {return py::array_t<double>(2,j.axes);},py::return_value_policy::reference_internal)
        .def_property_readonly("button",[](const joydata& j) {return py::array_t<int>(14,j.button);},py::return_value_policy::reference_internal);
    py::enum_<ControlMode>(m,"ControlMode")
        .value("USERMODE",ControlMode::USERMODE)   
        .value("DEFAULT",ControlMode::DEFAULT);   
    py::class_<Controllerbase>(m, "Controllerbase")
        .def(py::init<>())
        .def("init", &Controllerbase::init)
        .def("loadModel", &Controllerbase::loadModel)
        .def("get_joint_state", &Controllerbase::get_joint_state)
        .def("set_joint", &Controllerbase::set_joint)
        .def("get_jsdata", &Controllerbase::get_jsdata)
        .def("get_imu_data", &Controllerbase::get_imu_data)
        .def("start", &Controllerbase::start)
        .def("setcallback", &Controllerbase::setcallback)
        .def("setpycallback", &Controllerbase::setpycallback);
        
}