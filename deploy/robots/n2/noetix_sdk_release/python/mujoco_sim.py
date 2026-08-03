import mujoco, mujoco_viewer
import numpy as np
from walk_user_sim import HumanoidController

from scipy.spatial.transform import Rotation as R

class MujocoSim(HumanoidController):
    def __init__(self, mujoco_model_path, policy_model_path):    
        super().__init__(policy_model_path)

        self.mujoco_model_path = mujoco_model_path
        self.policy_model_path = policy_model_path

        self.dt = 0.001
        self.pos = [0.0, 0.0, 0.75]
        self.orientation = [1, 0, 0, 0]
        
        self.dof_pos = self.config.default_joint_angles
        print(self.dof_pos)
        self.init_qpos = self.pos + self.orientation + self.dof_pos
        
        self._init_mujoco()


    def _init_mujoco(self):
        print(self.mujoco_model_path)
        self.model = mujoco.MjModel.from_xml_path(self.mujoco_model_path)
        self.model.opt.timestep = self.dt
        self.data = mujoco.MjData(self.model)
        self.data.qpos = self.init_qpos
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)


    def get_obs(self):

        self.qpos = self.data.qpos.astype(np.double)[-18:] 
        # print(self.qpos)
        self.qvel = self.data.qvel.astype(np.double)[-18:]
        # print(self.data.qpos.astype(np.double)[[4,5,6,3]])
        # print(self.data.sensor('orientation'))
        
        quat = self.data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
        # quat =  self.data.qpos.astype(np.double)[[4,5,6,3]]
        # r = R.from_quat(quat)
        # v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
        omega = self.data.sensor('angular-velocity').data.astype(np.double)
        # gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
        q, dq = np.zeros(10), np.zeros(10)
        j = 0
        for i in range(18):
            if 4 <= i <= 8 or 13 <= i <= 17:
                q[j], dq[j] = self.qpos[i] , self.qvel[i]
                j += 1

        cmd = [0.6, 0, 0]
        # print(q, dq)
        return q, dq, quat, omega, cmd
        
 
    def ctrl(self):
        target_q = np.zeros((1,1,18), dtype=np.double)
        action = self.action * self.config.action_scale
        
        for i in range(18):
            if i < 4 or 9 <= i < 13:
                target_q[0,0,i] = 0
            elif 4 <= i < 9:
                target_q[0,0,i] =  action[0,0,i-4]
            elif 13<= i:
                target_q[0,0,i] =  action[0,0,i-8]
        
        target_dq = np.zeros((18), dtype=np.double)
        # print(target_q + self.dof_pos, )
        tau = (target_q + self.dof_pos - self.qpos) * self.config.kps + (target_dq - self.qvel) * self.config.kds
        tau = np.clip(tau, -self.config.tau_limit, self.config.tau_limit)  # Clamp torques
      
        self.data.ctrl = tau
        mujoco.mj_step(self.model, self.data)
        self.viewer.render()

    def run_mujoco(self):
        self.kps = np.array([30,30,30,30,90, 90, 120, 120, 15, 
                             30,30,30,30,90, 90, 120, 120, 15], dtype=np.double)
        
        self.kds = np.array([2,2,1,2,4, 4, 5, 5, 1, 
                             2,2,1,2,4, 4, 5, 5, 1], dtype=np.double)

        self.config.decimation = 10
        # self.kps, self.kds = np.zeros(18), np.zeros(18)
        # for i in range(18):
        #     self.kps[i], self.kds[i] = self.config.stiffness[i], self.config.damping[i]
        # self.mode_walk()

        while True:
            self.mode_walk()
            # time.sleep(0.00001)


if __name__ == '__main__':
    policy_model_path = '/home/oem/work/n2/n2-sdk/ning/policy_walk.onnx'
    mujoco_model_path = '/home/oem/work/n2/n2-sdk/robot/mjcf/N2.xml'
    sim = MujocoSim(mujoco_model_path, policy_model_path)
    sim.run_mujoco()



