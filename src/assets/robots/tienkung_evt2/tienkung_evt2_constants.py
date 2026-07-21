"""Tienkung EVT2 robot constants.

This file defines the EntityCfg for the Tienkung EVT2 humanoid robot.
Actuator parameters are ported from src/assets/robots/EVT2/tiangong.py.
Each unique (stiffness, damping, effort_limit) combination gets its own
BuiltinPositionActuatorCfg to match the per-joint values in tiangong.py.

Note: armature values are grouped by joint size (hip/knee large, waist/ankle/
shoulder pitch/elbow medium, shoulder roll/yaw small) because exact rotor inertia
and gear ratios are not available in the EVT2 asset files.
"""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

TIENKUNG_EVT2_XML: Path = (
    SRC_PATH / "assets" / "robots" / "tienkung_evt2" / "xmls" / "tiangong2dex_torq.xml"
)
assert TIENKUNG_EVT2_XML.exists(), f"MJCF not found: {TIENKUNG_EVT2_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, TIENKUNG_EVT2_XML.parent / "assets", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(TIENKUNG_EVT2_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Actuator config.
#
# Values below are taken from src/assets/robots/EVT2/tiangong.py.
# armature is set per joint size group (large/medium/small) because exact motor
# rotor inertia and gear ratios are unavailable.
##

# Legs.
HIP_YAW_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*hip_yaw.*_joint",),
    stiffness=150.0,
    damping=5.0,
    effort_limit=142.0,
    armature=0.015,
)

HIP_ROLL_PITCH_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*hip_roll.*_joint",
        ".*hip_pitch.*_joint",
    ),
    stiffness=300.0,
    damping=10.0,
    effort_limit=200.0,
    armature=0.03,
)

KNEE_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*knee_pitch.*_joint",),
    stiffness=330.0,
    damping=10.0,
    effort_limit=330.0,
    armature=0.03,
)

# Feet.
ANKLE_PITCH_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*ankle_pitch.*_joint",),
    stiffness=30.0,
    damping=2.5,
    effort_limit=100.0,
    armature=0.015,
)

ANKLE_ROLL_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*ankle_roll.*_joint",),
    stiffness=16.8,
    damping=1.4,
    effort_limit=50.0,
    armature=0.015,
)

# Waist.
WAIST_YAW_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=("waist_yaw_joint",),
    stiffness=400.0,
    damping=5.0,
    effort_limit=91.0,
    armature=0.015,
)

WAIST_ROLL_PITCH_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "waist_roll_joint",
        "waist_pitch_joint",
    ),
    stiffness=400.0,
    damping=10.0,
    effort_limit=91.0,
    armature=0.015,
)

# Head.
HEAD_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "head_yaw_joint",
        "head_pitch_joint",
    ),
    stiffness=10.0,
    damping=0.5,
    effort_limit=6.3,
    armature=0.005,
)

# Arms.
SHOULDER_PITCH_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*shoulder_pitch.*_joint",),
    stiffness=150.0,
    damping=5.0,
    effort_limit=90.0,
    armature=0.015,
)

SHOULDER_ROLL_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*shoulder_roll.*_joint",),
    stiffness=50.0,
    damping=2.5,
    effort_limit=60.0,
    armature=0.005,
)

SHOULDER_YAW_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*shoulder_yaw.*_joint",),
    stiffness=50.0,
    damping=2.5,
    effort_limit=36.0,
    armature=0.005,
)

ELBOW_PITCH_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*elbow_pitch.*_joint",),
    stiffness=150.0,
    damping=5.0,
    effort_limit=60.0,
    armature=0.015,
)

ELBOW_YAW_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*elbow_yaw.*_joint",),
    stiffness=50.0,
    damping=5.0,
    effort_limit=25.0,
    armature=0.005,
)

WRIST_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*wrist_pitch.*_joint",
        ".*wrist_roll.*_joint",
    ),
    stiffness=20.0,
    damping=2.0,
    effort_limit=25.0,
    armature=0.005,
)

##
# Keyframe config.
#
# Initial standing pose. Values are taken from tiangong.py init_state.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.97),
    joint_pos={
        # Legs.
        "hip_yaw_l_joint": 0.0,
        "hip_roll_l_joint": 0.0,
        "hip_pitch_l_joint": -0.15,
        "knee_pitch_l_joint": 0.3,
        "ankle_pitch_l_joint": -0.15,
        "ankle_roll_l_joint": 0.0,
        "hip_yaw_r_joint": 0.0,
        "hip_roll_r_joint": 0.0,
        "hip_pitch_r_joint": -0.15,
        "knee_pitch_r_joint": 0.3,
        "ankle_pitch_r_joint": -0.15,
        "ankle_roll_r_joint": 0.0,
        # Waist.
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "waist_pitch_joint": 0.0,
        # Head.
        "head_yaw_joint": 0.0,
        "head_pitch_joint": 0.0,
        # Arms.
        "shoulder_pitch_l_joint": 0.2,
        "shoulder_roll_l_joint": 0.1,
        "shoulder_yaw_l_joint": 0.0,
        "elbow_pitch_l_joint": -0.5,
        "elbow_yaw_l_joint": 0.0,
        "wrist_pitch_l_joint": 0.0,
        "wrist_roll_l_joint": 0.0,
        "shoulder_pitch_r_joint": 0.2,
        "shoulder_roll_r_joint": -0.1,
        "shoulder_yaw_r_joint": 0.0,
        "elbow_pitch_r_joint": -0.5,
        "elbow_yaw_r_joint": 0.0,
        "wrist_pitch_r_joint": 0.0,
        "wrist_roll_r_joint": 0.0,
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
#
# Enables collisions for all geoms named *_collision. Feet (ankle pitch/roll
# links) are given condim=3 for stable contact; the rest use condim=1.
##

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={
        r".*ankle_(pitch|roll)_[lr]_link_collision$": 3,
        ".*_collision": 1,
    },
    priority={
        r".*ankle_(pitch|roll)_[lr]_link_collision$": 1,
    },
    friction={
        r".*ankle_(pitch|roll)_[lr]_link_collision$": (0.6,),
    },
)

##
# Final config.
##

TIENKUNG_EVT2_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        HIP_YAW_ACTUATOR,
        HIP_ROLL_PITCH_ACTUATOR,
        KNEE_ACTUATOR,
        ANKLE_PITCH_ACTUATOR,
        ANKLE_ROLL_ACTUATOR,
        WAIST_YAW_ACTUATOR,
        WAIST_ROLL_PITCH_ACTUATOR,
        HEAD_ACTUATOR,
        SHOULDER_PITCH_ACTUATOR,
        SHOULDER_ROLL_ACTUATOR,
        SHOULDER_YAW_ACTUATOR,
        ELBOW_PITCH_ACTUATOR,
        ELBOW_YAW_ACTUATOR,
        WRIST_ACTUATOR,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_tienkung_evt2_robot_cfg() -> EntityCfg:
    """Get a fresh Tienkung EVT2 robot configuration instance."""
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=TIENKUNG_EVT2_ARTICULATION,
    )


##
# Action scale.
#
# Computes per-regex action scale as 0.25 * effort_limit / stiffness.
##

TIENKUNG_EVT2_ACTION_SCALE: dict[str, float] = {}
for a in TIENKUNG_EVT2_ARTICULATION.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg)
    e = a.effort_limit
    s = a.stiffness
    names = a.target_names_expr
    assert e is not None
    for n in names:
        TIENKUNG_EVT2_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    robot = Entity(get_tienkung_evt2_robot_cfg())
    viewer.launch(robot.spec.compile())
