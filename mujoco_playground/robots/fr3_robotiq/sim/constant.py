"""Joint and actuator names for the FR3 + Robotiq 2F-85 model."""

ATTACHMENT_PREFIX = "robot/"
GRIPPER_PREFIX = "robotiq/"
EE_SITE = f"{ATTACHMENT_PREFIX}{GRIPPER_PREFIX}pinch"

ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
GRIPPER_JOINT = "right_driver_joint"
JOINTS = [f"{ATTACHMENT_PREFIX}{joint}" for joint in ARM_JOINTS] + [
    f"{ATTACHMENT_PREFIX}{GRIPPER_PREFIX}{GRIPPER_JOINT}"
]

ARM_ACTUATORS = ARM_JOINTS.copy()
GRIPPER_ACTUATOR = f"{GRIPPER_PREFIX}fingers_actuator"
ACTUATORS = [f"{ATTACHMENT_PREFIX}{name}" for name in ARM_ACTUATORS] + [
    f"{ATTACHMENT_PREFIX}{GRIPPER_ACTUATOR}"
]

# Menagerie's FR3 home keyframe, with the Robotiq driver fully open.
HOME_QPOS = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853, 0.0]
HOME_CTRL = HOME_QPOS[:7] + [0.0]

# Policy outputs in [-1, 1] move the arm around home and map the gripper to
# the Robotiq actuator's [0, 255] control range.
ACTION_SCALE = [0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 127.5]
ACTION_OFFSET = HOME_QPOS[:7] + [127.5]
