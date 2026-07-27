from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  noetix_n2_flat_env_cfg,
  noetix_n2_rough_env_cfg,
)
from .rl_cfg import noetix_n2_ppo_runner_cfg

register_mjlab_task(
  task_id="Noetix-N2-Rough",
  env_cfg=noetix_n2_rough_env_cfg(),
  play_env_cfg=noetix_n2_rough_env_cfg(play=True),
  rl_cfg=noetix_n2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Noetix-N2-Flat",
  env_cfg=noetix_n2_flat_env_cfg(),
  play_env_cfg=noetix_n2_flat_env_cfg(play=True),
  rl_cfg=noetix_n2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
