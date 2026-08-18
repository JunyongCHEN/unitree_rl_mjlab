"""从 mjlab checkpoint 导出 sim2sim 可用的 policy.pt"""
import sys
from dataclasses import asdict

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import src.tasks  # 注册 Noetix-N2-Flat


def export_jit(task_id: str, checkpoint: str, out_dir: str):
    device = "cpu"
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = 1
    agent_cfg = load_rl_cfg(task_id)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id)
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)

    runner.export_policy_to_jit(out_dir, filename="policy.pt")
    print(f"exported JIT -> {out_dir}/policy.pt")
    env.close()


if __name__ == "__main__":
    task_id = sys.argv[1] if len(sys.argv) > 1 else "Noetix-N2-Flat"
    checkpoint = sys.argv[2]  # logs/rsl_rl/noetix_n2_velocity/<ts>/model_1000.pt
    out_dir = sys.argv[3] if len(sys.argv) > 3 else checkpoint.rsplit("/", 1)[0]
    export_jit(task_id, checkpoint, out_dir)