"""Deployment-shaped sim2sim for the mjlab-trained Noetix N2 velocity policy.

This runs the exported ``policy.onnx`` through **onnxruntime** (the same engine
used on the robot) inside a hand-written 50 Hz control loop that rebuilds the
*exact* mjlab observation from raw robot state. It is meant as the reference
implementation to port onto the real N2 via ``noetix_sdk_n2``:

    * ``read_robot_state()``  -> replace with SDK IMU + joint-encoder reads.
    * ``apply_joint_targets()`` -> replace with SDK PD / torque command.

Everything else (observation layout, scaling, joint order, PD gains, default
pose, action scale, control rate) is read from the ONNX metadata that
``VelocityOnPolicyRunner`` attaches at export time, so it stays in sync with the
trained policy.

Observation layout (65), matching src/tasks/velocity/velocity_env_cfg.py (Flat):
    base_ang_vel(3) | projected_gravity(3) | command(3) | phase(2)
    | joint_pos - default(18) | joint_vel(18) | last_action(18)

Usage (headless -> writes an mp4; add --live with a display for a window):
    python scripts/n2_sim2sim.py \
        --onnx logs/rsl_rl/noetix_n2_velocity/<ts>/policy.onnx \
        --command 0.4 0.0 0.0 --duration 10 --video /tmp/n2_sim2sim.mp4
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation

from mjlab.entity.entity import Entity
from src.assets.robots.noetix_n2.noetix_n2_constants import get_noetix_n2_robot_cfg

CONTROL_DECIMATION = 4  # policy @ 50 Hz, physics @ 200 Hz (mjlab Flat: dt=0.005)
SIM_DT = 0.005
PHASE_PERIOD = 0.6  # matches mdp.phase(period=0.6)
BASE_INIT_HEIGHT = 0.755


def load_policy(onnx_path: str):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    md = sess.get_modelmeta().custom_metadata_map
    joint_names = md["joint_names"].split(",")
    kp = np.array([float(x) for x in md["joint_stiffness"].split(",")], dtype=np.float32)
    kd = np.array([float(x) for x in md["joint_damping"].split(",")], dtype=np.float32)
    default_q = np.array([float(x) for x in md["default_joint_pos"].split(",")], dtype=np.float32)
    action_scale = np.array([float(x) for x in md["action_scale"].split(",")], dtype=np.float32)
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    return sess, in_name, out_name, joint_names, kp, kd, default_q, action_scale


def build_model() -> mujoco.MjModel:
    """Compile the mjlab N2 spec (actuators + collision) and add a ground plane."""
    spec = Entity(get_noetix_n2_robot_cfg()).spec
    spec.worldbody.add_light(pos=[0, 0, 5], dir=[0, 0, -1],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 0.05]
    floor.condim = 3
    floor.friction = [1.0, 0.005, 0.0001]
    floor.rgba = [0.5, 0.55, 0.6, 1.0]
    model = spec.compile()
    model.opt.timestep = SIM_DT
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="Path to exported policy.onnx")
    ap.add_argument("--command", type=float, nargs=3, default=[0.4, 0.0, 0.0],
                    metavar=("VX", "VY", "WZ"), help="Desired [vx, vy, wz]")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--video", type=str, default="/tmp/n2_sim2sim.mp4")
    args = ap.parse_args()

    sess, in_name, out_name, joint_names, kp, kd, default_q, action_scale = load_policy(args.onnx)
    n = len(joint_names)
    model = build_model()
    data = mujoco.MjData(model)

    # Index maps: policy joint order -> MuJoCo qpos/qvel/actuator indices.
    qpos_adr = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joint_names])
    dof_adr = np.array([model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joint_names])
    act_id = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_pos")
                       if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{j}_pos") >= 0
                       else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in joint_names])
    # mjlab names position actuators after the joint; fall back to trnid match.
    if np.any(act_id < 0):
        act_id = np.array([next(a for a in range(model.nu) if model.actuator_trnid[a, 0] ==
                                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)) for j in joint_names])
    ang_vel_sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    ang_vel_adr = model.sensor_adr[ang_vel_sensor]
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    # Initial pose = default standing pose.
    data.qpos[:3] = [0.0, 0.0, BASE_INIT_HEIGHT]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qpos_adr] = default_q
    mujoco.mj_forward(model, data)

    command = np.array(args.command, dtype=np.float32)
    last_action = np.zeros(n, dtype=np.float32)

    # ---- functions to REPLACE on real hardware (noetix_sdk_n2) ----
    def read_robot_state():
        q = data.qpos[qpos_adr].astype(np.float32)                       # joint pos (rad)
        dq = data.qvel[dof_adr].astype(np.float32)                       # joint vel (rad/s)
        ang_vel = data.sensordata[ang_vel_adr:ang_vel_adr + 3].astype(np.float32)  # base gyro
        quat_wxyz = data.qpos[3:7]                                       # base orientation
        grav_b = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).apply([0, 0, -1], inverse=True)
        return q, dq, ang_vel, grav_b.astype(np.float32)

    def apply_joint_targets(target_q):
        # Sim uses mjlab position actuators (kp/kd baked in). On the robot:
        #   tau = kp * (target_q - q) - kd * dq   (or the SDK's position mode w/ kp,kd)
        data.ctrl[act_id] = target_q
    # ---------------------------------------------------------------

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 3.0, 120.0, -12.0

    import imageio
    frames = []
    n_steps = int(args.duration / SIM_DT)
    control_step = 0
    for i in range(n_steps):
        if i % CONTROL_DECIMATION == 0:
            q, dq, ang_vel, grav_b = read_robot_state()
            t = control_step * SIM_DT * CONTROL_DECIMATION
            gp = (t % PHASE_PERIOD) / PHASE_PERIOD
            if np.linalg.norm(command) < 0.1:
                phase = np.zeros(2, dtype=np.float32)
            else:
                phase = np.array([np.sin(2 * np.pi * gp), np.cos(2 * np.pi * gp)], dtype=np.float32)
            obs = np.concatenate([
                ang_vel,               # base_ang_vel (3)
                grav_b,                # projected_gravity (3)
                command,               # command (3)
                phase,                 # phase (2)
                q - default_q,         # joint_pos_rel (18)
                dq,                    # joint_vel (18)
                last_action,           # last_action (18)
            ]).astype(np.float32)[None]
            action = sess.run([out_name], {in_name: obs})[0][0]
            if not np.all(np.isfinite(action)):
                print(f"[ERR] non-finite action at step {i}"); break
            last_action = action
            target_q = default_q + action_scale * action
            apply_joint_targets(target_q)
            control_step += 1

        mujoco.mj_step(model, data)

        if i % CONTROL_DECIMATION == 0:
            cam.lookat = data.xpos[base_bid]
            renderer.update_scene(data, cam)
            frames.append(renderer.render())

    base_h = data.xpos[base_bid][2]
    print(f"finished {control_step} control steps; final base height = {base_h:.3f} m "
          f"(fell if << {BASE_INIT_HEIGHT:.2f})")
    imageio.mimsave(args.video, frames, fps=int(1.0 / (SIM_DT * CONTROL_DECIMATION)), macro_block_size=1)
    print("wrote", args.video)


if __name__ == "__main__":
    main()
