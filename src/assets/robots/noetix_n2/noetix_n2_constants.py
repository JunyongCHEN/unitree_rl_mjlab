"""Noetix N2 robot constants.

Defines the EntityCfg for the Noetix N2, an 18-DoF humanoid
(5 DoF/leg: hip yaw/roll/pitch, knee, ankle-pitch; 4 DoF/arm: shoulder
pitch/roll/yaw, elbow).

Parameter sources (highest priority first):
  - PD gains, default pose and action scale: noetix_n2_gym
    (humanoid/envs/n2/n2_config.py and sim2sim/configs/n2_18dof.yaml).
  - Effort limits: N2 URDF joint <limit effort=...>.
  - Masses/inertias: taken as-is from the source MJCF (all bodies have an
    explicit <inertial>; total mass ~33.2 kg).
  - armature/frictionloss: from the source MJCF <default> (0.01 / 0.1).
    These are the model's own values; exact rotor inertia / gear ratios are
    unavailable, so armature is a documented estimate.
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

NOETIX_N2_XML: Path = (
    SRC_PATH / "assets" / "robots" / "noetix_n2" / "xmls" / "n2.xml"
)
assert NOETIX_N2_XML.exists(), f"MJCF not found: {NOETIX_N2_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, NOETIX_N2_XML.parent / "assets", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(NOETIX_N2_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Actuator config.
#
# stiffness (kp) / damping (kd) are the deployment PD gains from
# noetix_n2_gym (n2_config.py control.stiffness/damping, matching
# sim2sim/configs/n2_18dof.yaml kps/kds). effort_limit is the URDF joint
# effort. armature/frictionloss follow the source MJCF defaults.
##

# Arms: shoulder pitch/roll/yaw + elbow. kp=30, kd=1, effort=27 Nm.
ARM_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_arm_.*_joint",),
    stiffness=30.0,
    damping=1.0,
    effort_limit=27.0,
    armature=0.01,
    frictionloss=0.1,
)

# Hip yaw + roll. kp=80, kd=5, effort=90 Nm.
HIP_YAW_ROLL_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_leg_hip_yaw_joint",
        ".*_leg_hip_roll_joint",
    ),
    stiffness=80.0,
    damping=5.0,
    effort_limit=90.0,
    armature=0.01,
    frictionloss=0.1,
)

# Hip pitch + knee. kp=120, kd=5, effort=150 Nm.
HIP_PITCH_KNEE_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_leg_hip_pitch_joint",
        ".*_leg_knee_joint",
    ),
    stiffness=120.0,
    damping=5.0,
    effort_limit=150.0,
    armature=0.01,
    frictionloss=0.1,
)

# Ankle pitch. kp=20, kd=2, effort=70 Nm.
ANKLE_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_leg_ankle_joint",),
    stiffness=20.0,
    damping=2.0,
    effort_limit=70.0,
    armature=0.01,
    frictionloss=0.1,
)

##
# Keyframe config.
#
# Standing pose = the deployment default from noetix_n2_gym
# (default_joint_angles). The pelvis height (0.755 m) is set so the foot
# contact capsules just touch the ground plane at this pose.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.755),
    joint_pos={
        # Arms.
        ".*_arm_shoulder_pitch_joint": 0.0,
        "L_arm_shoulder_roll_joint": 0.2,
        "R_arm_shoulder_roll_joint": -0.2,
        ".*_arm_shoulder_yaw_joint": 0.0,
        ".*_arm_elbow_joint": 0.0,
        # Legs (mild crouch).
        ".*_leg_hip_yaw_joint": 0.0,
        ".*_leg_hip_roll_joint": 0.0,
        ".*_leg_hip_pitch_joint": -0.1495,
        ".*_leg_knee_joint": 0.3215,
        ".*_leg_ankle_joint": -0.1720,
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
#
# Enables collisions for all *_collision geoms. Feet (4 capsules/foot) use
# condim=3 with softened contact and low torsion/roll friction for stable
# ground contact; all other body primitives use condim=1.
##

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={
        r"^(left|right)_foot[1-4]_collision$": 3,
        ".*_collision": 1,
    },
    priority={
        r"^(left|right)_foot[1-4]_collision$": 1,
    },
    friction={
        r"^(left|right)_foot[1-4]_collision$": (0.6, 0.005, 0.0001),
    },
    solref={
        r"^(left|right)_foot[1-4]_collision$": (0.02, 1.0),
    },
)

##
# Final config.
##

NOETIX_N2_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        ARM_ACTUATOR,
        HIP_YAW_ROLL_ACTUATOR,
        HIP_PITCH_KNEE_ACTUATOR,
        ANKLE_ACTUATOR,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_noetix_n2_robot_cfg() -> EntityCfg:
    """Get a fresh Noetix N2 robot configuration instance."""
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=NOETIX_N2_ARTICULATION,
    )


##
# Action scale.
#
# noetix_n2_gym uses a constant action_scale of 0.25 rad for all joints
# (n2_config.py control.action_scale, sim2sim action_scale). Keep that.
##

NOETIX_N2_ACTION_SCALE: dict[str, float] = {}
for a in NOETIX_N2_ARTICULATION.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg)
    for n in a.target_names_expr:
        NOETIX_N2_ACTION_SCALE[n] = 0.25


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    robot = Entity(get_noetix_n2_robot_cfg())
    viewer.launch(robot.spec.compile())
