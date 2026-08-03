import numpy as np
import ctypes
import time
import sys
import onnxruntime as ort
import copy
import multiprocessing
import math
from collections import deque
import onnx
from enum import Enum

sys.path.append("./")
import noetix_interface


class RobotConfig:
    default_joint_angles = [
        0.0, 0.3, 0.0, 0.0,
        0.0, 0.0, -0.1495, 0.3215, -0.1720,
        0.0, -0.3, 0.0, 0.0,
        0.0, 0.0, -0.1495, 0.3215, -0.1720
    ]
    walkdefault_joint_angles = [
        0.0, 0.0, -0.1495, 0.3215, -0.1720,
        0.0, 0.0, -0.1495, 0.3215, -0.1720
    ]


    kps = np.array([30.0, 30.0, 30.0, 30.0,
        90.0, 90.0, 120.0, 120.0, 15.0,
        30.0, 30.0, 30.0, 30.0,
        90.0, 90.0, 120.0, 120.0, 15.0], dtype=np.double)

    kds = np.array([ 2.0, 2.0, 1.0, 2.0,
        4.0, 4.0, 5.0, 5.0, 1.0,
        2.0, 2.0, 1.0, 2.0,
        4.0, 4.0, 5.0, 5.0, 1.0], dtype=np.double)

    stiffness = [
        30.0, 30.0, 30.0, 30.0,
        90.0, 90.0, 120.0, 120.0, 15.0,
        30.0, 30.0, 30.0, 30.0,
        90.0, 90.0, 120.0, 120.0, 15.0
    ]
    damping = [
        2.0, 2.0, 1.0, 2.0,
        4.0, 4.0, 5.0, 5.0, 1.0,
        2.0, 2.0, 1.0, 2.0,
        4.0, 4.0, 5.0, 5.0, 1.0
    ]
    tau_limit = np.array([36, 36, 36, 36,
                        90, 90, 150, 150, 36,
                        36, 36, 36, 36, 
                        90, 90, 150, 150, 36], dtype=np.double)


    action_scale = 0.25
    # 10 for sim; 5 for real
    decimation = 10 
    cycle_time = 1.0
    clip_actions = 18
    clip_observations = 18
    lin_vel = 2
    ang_vel = 1
    dof_pos = 1
    dof_vel = 0.05
    quat = 1
    action_size = 10
    observation_size = 38
    stack_size = 20
    scalex = 1.0
    scaley = 0.3
    scalez = 0.8


class HumanoidController:
    def __init__(self, model_path):
        self.noetix = noetix_interface.Controllerbase()
        self.config = RobotConfig()
        self.model_path = model_path
        self.policy = self.load_model(model_path)
        self.action = np.zeros((self.config.action_size), dtype=np.double)
        self.lastaction = np.zeros((self.config.action_size), dtype=np.double)
        self.hist_obs = deque()
        for _ in range(self.config.stack_size):
            self.hist_obs.append(np.zeros([1, self.config.observation_size], dtype=np.double))
        self.count_lowlevel = 0
    
    def quaternion_to_euler_array(self, quat):
        # Ensure quaternion is in the correct format [x, y, z, w]
        x, y, z, w = quat
        
        # Roll (x-axis rotation)
        t0 = +2.0 * (w * x + y * z)
        t1 = w*w - x*x - y*y + z*z
        roll_x = np.arctan2(t0, t1)
        
        # Pitch (y-axis rotation)
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.arcsin(t2)
        
        # Yaw (z-axis rotation)
        t3 = +2.0 * (w * z + x * y)
        t4 = w*w + x*x - y*y - z*z
        yaw_z = np.arctan2(t3, t4)
        
        # Returns roll, pitch, yaw in a NumPy array in radians
        return np.array([roll_x, pitch_y, yaw_z])

    def load_model(self, path):
        model = onnx.load(path)
        onnx.checker.check_model(model)
        return ort.InferenceSession(path)

    def get_obs(self):
        joint_state = self.noetix.get_joint_state()
        imu_data = self.noetix.get_imu_data()
        q, dq, quat = np.zeros(10), np.zeros(10), imu_data.ori
        j = 0
        for i in range(18):
            if 4 <= i <= 8 or 13 <= i <= 17:
                q[j], dq[j] = joint_state[i].pos, joint_state[i].vel
                j += 1
        
        remote_data = self.noetix.get_jsdata()
        axes = remote_data.axes
        vx = np.clip(axes[1] * self.config.scalex, -0.8, 0.5)
        dyaw = axes[0] * self.config.scaley
        cmd = [vx, 0, dyaw]
        return q, dq, quat, imu_data.angular_vel, cmd


    def compute_observation(self):
        obs = np.zeros([1, self.config.observation_size], dtype=np.float32)
        q, dq, quat, omega, cmd = self.obs
        eu_ang = self.quaternion_to_euler_array(quat)
        eu_ang[eu_ang > np.pi] -= 2 * np.pi
        obs[0, :3] = [cmd[0] * self.config.lin_vel, cmd[1], cmd[2] * self.config.ang_vel]
        obs[0, 3:6] = omega * self.config.ang_vel
        obs[0, 6:8] = eu_ang[:2] * self.config.quat
        obs[0, 8:18] = (q - np.array(self.config.walkdefault_joint_angles)) * self.config.dof_pos
        obs[0, 18:28] = dq * self.config.dof_vel
        obs[0, 28:38] = self.lastaction
        obs = np.clip(obs, -self.config.clip_observations, self.config.clip_observations)
        self.hist_obs.append(obs)
        self.hist_obs.popleft()

        policy_input = np.zeros([1, self.config.observation_size*self.config.stack_size], dtype=np.float32)
        for i in range(self.config.stack_size):
            policy_input[0, i * self.config.observation_size: (i + 1) * self.config.observation_size] = self.hist_obs[i][0, :]
        return policy_input

    def compute_actions(self, policy_input):
        action = self.policy.run(None, {"policy_input": policy_input.astype(np.float32)})
        return np.clip(action, -self.config.clip_actions, self.config.clip_actions)
    
    def mode_walk(self):
        self.obs = self .get_obs()
        if self.count_lowlevel % self.config.decimation == 0:
            policy_input = self.compute_observation()
            self.action = self.compute_actions(policy_input)
            
            self.lastaction = self.action

        self.ctrl()

        self.count_lowlevel += 1

    def ctrl(self):
        motorcmd = noetix_interface.MotorCmd()
        for i in range(self.config.action_size):
            if i <5:
                j= i+4
            if i > 4:
                j = i+8
            pos_des = self.action[0,0,i] * self.config.action_scale+self.config.default_joint_angles[j]
            stiffness = self.config.stiffness[j]
            damping = self.config.damping[j]
            motorcmd.pos = pos_des
            motorcmd.kp = stiffness
            motorcmd.kd = damping 
            motorcmd.motor_id = j
            motorcmd.vel = 0
            motorcmd.tau = 0
            self.noetix.set_joint(motorcmd)
        for i in range(8):
            if i <4:
                j= i
            if i >= 4:
                j = i+5
            joint_state = self.noetix.get_joint_state()
            cur_pos = joint_state[j].pos -self.config.default_joint_angles[j]
            pos_des = 0.75 * cur_pos + 0.25 * self.config.default_joint_angles[j]
            
            stiffness = self.config.stiffness[j]
            damping = self.config.damping[j]
            motorcmd.pos = pos_des
            motorcmd.kp = stiffness
            motorcmd.kd = damping 
            motorcmd.motor_id = j
            motorcmd.vel = 0
            motorcmd.tau = 0
            self.noetix.set_joint(motorcmd)    
        for i in range(self.config.action_size):
            j = i + 4 if i < 5 else i + 8
            motorcmd.pos = self.action[0, 0, i] * self.config.action_scale + self.config.default_joint_angles[j]
            motorcmd.kp = self.config.stiffness[j]
            motorcmd.motor_id = j
            motorcmd.vel, motorcmd.tau = 0, 0
            self.noetix.set_joint(motorcmd)

    def my_callback(self):
        self.mode_walk()

    def run(self):
        self.noetix.setpycallback(self.my_callback)
        self.noetix.init(noetix_interface.ControlMode.USERMODE)
        while True:
            time.sleep(0.00001)


if __name__ == '__main__':
    model_path = '/home/oem/work/n2/n2-sdk/ning/policy_walk.onnx'
    controller = HumanoidController(model_path)
    controller.run()
