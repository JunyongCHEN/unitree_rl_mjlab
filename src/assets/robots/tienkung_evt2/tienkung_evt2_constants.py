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
# Initial standing pose: arm angles follow the tiangong.py reference, while the
# legs are bent into a deeper crouch to lower the COM (see HOME_KEYFRAME).
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    # Crouched standing pose matched to the Unitree H1 configuration. Bending the
    # knees and ankles more lowers the COM and makes the initial state easier to
    # balance under MuJoCo's position actuators. The pelvis height is set so the
    # foot capsule geoms just touch the ground plane.
    pos=(0.0, 0.0, 0.97),
    joint_pos={
        # Legs.
        ".*hip_pitch.*_joint": -0.2,
        ".*knee_pitch.*_joint": 0.5,
        ".*ankle_pitch.*_joint": -0.3,
        # Arms. Values follow the TienKung-Lab reference (tiangong.py) and the
        # model's actual joint conventions. IMPORTANT: elbow_pitch range is
        # [-2.618, +0.262]; its natural flexion direction is NEGATIVE, so the
        # forearm bends forward at -0.5. A positive elbow default (e.g. the old
        # +0.52) is past the joint limit and pins the arm straight out behind
        # the robot. shoulder_roll splays the arms slightly out from the torso.
        ".*shoulder_pitch.*_joint": 0.2,
        "shoulder_roll_l_joint": 0.1,
        "shoulder_roll_r_joint": -0.1,
        ".*elbow_pitch.*_joint": -0.5,
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
#
# Enables collisions for all geoms named *_collision. Feet (ankle pitch/roll
# links) are given condim=3 for stable contact; the rest use condim=1.
# Foot contact is softened (solref 0.02) and low torsion/slip friction to match
# the G1 foot setup and avoid huge impact forces/NaN when the robot falls.
##

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={
        r".*foot.*_collision$": 3,
        ".*_collision": 1,
    },
    priority={
        r".*foot.*_collision$": 1,
    },
    friction={
        r".*foot.*_collision$": (0.6, 0.005, 0.0001),
    },
    solref={
        r".*foot.*_collision$": (0.02, 1.0),
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
# The original TienKung-Lab dex_walk config uses a constant action scale of
# 0.25 rad for all joints. Keep that instead of the G1/H1 effort/stiffness
# scaling, which gives the EVT2 ankles too much authority and drives the
# heavy robot into joint limits.
##

TIENKUNG_EVT2_ACTION_SCALE: dict[str, float] = {}
for a in TIENKUNG_EVT2_ARTICULATION.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg)
    names = a.target_names_expr
    for n in names:
        TIENKUNG_EVT2_ACTION_SCALE[n] = 0.25


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    robot = Entity(get_tienkung_evt2_robot_cfg())
    viewer.launch(robot.spec.compile())
