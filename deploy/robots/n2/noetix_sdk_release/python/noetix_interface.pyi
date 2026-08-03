from typing import Tuple, overload
import numpy as np
import numpy.typing as npt
from enum import Enum


class JointState:
    arm_l1_joint: float
    arm_l2_joint: float
    arm_l3_joint: float
    arm_l4_joint: float
    leg_l1_joint: float
    leg_l2_joint: float
    leg_l3_joint: float
    leg_l4_joint: float
    leg_l5_joint: float
    arm_r1_joint: float
    arm_r2_joint: float
    arm_r3_joint: float
    arm_r4_joint: float
    leg_r1_joint: float
    leg_r2_joint: float
    leg_r3_joint: float
    leg_r4_joint: float
    leg_r5_joint: float
class MotorState:
    pos: float
    vel: float
    tau: float
    motor_id: uint16
    error: uint8
class MotorCmd:
    pos: float
    vel: float
    tau: float
    kp: float
    kd: float
    motor_id: uint16
class NingImuData:
    ori:  npt.NDArray[np.float64]
    ori_cov:  npt.NDArray[np.float64]
    angular_vel: npt.NDArray[np.float64]
    angular_vel_cov: npt.NDArray[np.float64]
    linear_acc: npt.NDArray[np.float64] 
    linear_acc_cov:npt.NDArray[np.float64]      
class joydata:
    axes:  npt.NDArray[np.float64]
    button:  npt.NDArray[np.int]
   
class Controllerbase:
    @overload
    def __init__(self, mode: int) -> bool: ...
    def loadModel(self) -> None: ...
    def set_joint(self, motorcmd: MotorCmd) -> None: ...
    def get_joint_state(self) -> np.ndarray[MotorState]: ...
    def get_jsdata(self) -> joydata: ...
    def get_imu_data(self) -> NingImuData: ...
    def start(self) -> None: ...
    def setpycallback(self,cb:ExternFuntionType) -> None: ...


