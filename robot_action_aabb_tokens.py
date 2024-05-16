import math

# 计算位置离散值
def to_location_token(value, min_val=0.0, max_val=1.0):
    normalized_value = (value - min_val) / (max_val - min_val)
    return f"<loc{round(normalized_value * 255)}>"

# 计算旋转角度离散值
def to_rotation_token(value, min_val=0.0, max_val=2 * math.pi):
    normalized_value = (value - min_val) / (max_val - min_val)
    return f"<arot{round(normalized_value * 255)}>"

# 计算夹持器状态
def to_gripper_token(open_status):
    return f"<gripper{int(open_status)}>"

# 机器人7自由度动作的完整示例
# 假设某机器人要完成以下动作：
# 位置坐标 (x, y, z) = (0.2, 0.4, 0.8)
# 旋转角度 (roll, pitch, yaw) = (30°, 45°, 60°)
# 夹持器状态 open_status = 1（关闭）

# 位置
loc_x = to_location_token(0.2)
loc_y = to_location_token(0.4)
loc_z = to_location_token(0.8)

# 旋转角度
roll_rad = math.radians(30)
pitch_rad = math.radians(45)
yaw_rad = math.radians(60)

rot_roll = to_rotation_token(roll_rad)
rot_pitch = to_rotation_token(pitch_rad)
rot_yaw = to_rotation_token(yaw_rad)

# 夹持器
gripper = to_gripper_token(1)

# 完整动作序列
action_sequence = f"{loc_x} {loc_y} {loc_z} {rot_roll} {rot_pitch} {rot_yaw} {gripper} <ACT SEP>"

# AABB位置示例
# 假设物体的边界框坐标如下：
# xmin = 0.2, ymin = 0.1, zmin = 0.3
# xmax = 0.7, ymax = 0.8, zmax = 0.9

aabb_tokens = [
    to_location_token(0.2),
    to_location_token(0.1),
    to_location_token(0.3),
    to_location_token(0.7),
    to_location_token(0.8),
    to_location_token(0.9),
]

aabb_sequence = " ".join(aabb_tokens)

# 打印完整动作序列和AABB序列
complete_example = f"Robotic Action Sequence: {action_sequence}\nAABB Tokens: {aabb_sequence}"
print(complete_example)
