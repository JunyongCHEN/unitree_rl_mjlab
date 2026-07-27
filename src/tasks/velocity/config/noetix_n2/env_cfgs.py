"""Noetix N2 velocity environment configurations.

Based on the Unitree G1 velocity config (a similar small ~33 kg humanoid) with
N2-specific mappings:
  - base body is ``base_link`` (single trunk; no separate pelvis/torso).
  - 4 foot capsules per foot (``{left,right}_foot[1-4]_collision``).
  - foot contact subtree is the ankle link (``{L,R}_leg_ankle_link``).
  - N2 has no waist/wrist joints and a single (pitch) ankle joint.
  - command curriculum is kept gentle because N2 has short feet
    (~5.5 cm fore-aft) and is harder to balance than G1.
"""

from src.assets.robots import (
  NOETIX_N2_ACTION_SCALE,
  get_noetix_n2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

# N2 base (trunk) body: used for terrain scan frame, viewer, orientation /
# angular-velocity rewards and COM randomization.
BASE_BODY = "base_link"


def noetix_n2_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Noetix N2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_noetix_n2_robot_cfg()}

  # Set raycast sensor frame to the N2 trunk.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = BASE_BODY

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 5)
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(L_leg_ankle_link|R_leg_ankle_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=BASE_BODY, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=BASE_BODY, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = NOETIX_N2_ACTION_SCALE

  cfg.viewer.body_name = BASE_BODY

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.0
  # N2 has short feet and is harder to balance; keep more envs standing and use
  # conservative velocity commands (close to the noetix_n2_gym command ranges).
  twist_cmd.rel_standing_envs = 0.15
  twist_cmd.ranges.lin_vel_x = (-0.8, 1.2)
  twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  # Gentler command curriculum than the shared default (which ramps to 2 m/s).
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (-0.5, 0.8), "lin_vel_y": (-0.4, 0.4), "ang_vel_z": (-1.0, 1.0)},
    {"step": 3000 * 24, "lin_vel_x": (-0.8, 1.2), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
  ]

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = (BASE_BODY,)

  # Posture std per joint. N2 has no waist/wrist joints and a single (pitch)
  # ankle. Knees/hip_pitch loosest for stride; hip roll/yaw tighter for lateral
  # stability; ankle moderate for foot clearance; arms moderate for swing.
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.5,
    r".*ankle.*": 0.15,
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.25,
    r".*knee.*": 0.5,
    r".*ankle.*": 0.25,
    r".*shoulder_pitch.*": 0.25,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = (BASE_BODY,)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (BASE_BODY,)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  # Modest alive bonus + softened termination penalty to help the short-footed
  # robot learn to balance before the huge default fall penalty dominates.
  cfg.rewards["is_terminated"].weight = -50.0
  cfg.rewards["alive"] = RewardTermCfg(
    func=envs_mdp.is_alive,
    weight=0.5,
    params={},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def noetix_n2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Noetix N2 flat terrain velocity configuration."""
  cfg = noetix_n2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.4, 0.4)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
