"""手动玩 Super Mario Bros 1-1 关卡，支持录制动作序列用于课程学习。

用法:
  python play-mario.py                     # 普通游玩
  python play-mario.py --record 2850       # 录制到 x=2850 的动作序列
  python play-mario.py --record-all        # 连续录制所有课程位置的动作序列

录制模式下:
  - 无敌状态，到达目标 x 位置后自动保存
  - 按 R 键可以重新开始录制
  - 动作序列保存到 curriculum_actions/target_XXXX.npy
"""
import sys
import os
import time
import argparse
import numpy as np

# 将 gym-super-mario-bros 加入 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gym-super-mario-bros"))

# 应用 NumPy 2.0 兼容补丁（修复 uint8 溢出）
import SASR.compat_patches  # noqa: F401

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros import SuperMarioBrosEnv
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from pyglet.window import key as pyglet_key

# 课程学习目标位置（按 x 从小到大排列，即游玩时依次经过的顺序）
CURRICULUM_POSITIONS = [
    (40, 79), 
    (500, 79),
    (1000, 79),
    (1500, 79),
    (1900, 79),
    (2700, 79),
]

# NES 按钮名称 → 位掩码
_BUTTON_TO_BIT = {
    'A': 1, 'B': 2, 'select': 4, 'start': 8,
    'up': 16, 'down': 32, 'left': 64, 'right': 128,
}


def buttons_to_bitmask(button_names):
    """将按钮名称列表转换为 NES 原始位掩码。"""
    mask = 0
    for name in button_names:
        if name != 'NOOP':
            mask |= _BUTTON_TO_BIT[name]
    return mask


# --------------- 键盘状态追踪 ---------------
keys_pressed = set()


def _on_key_press(symbol, modifiers):
    keys_pressed.add(symbol)


def _on_key_release(symbol, modifiers):
    keys_pressed.discard(symbol)


# pyglet 按键 → NES 按钮名称
KEY_MAP = {
    pyglet_key.RIGHT: 'right',
    pyglet_key.LEFT:  'left',
    pyglet_key.UP:    'up',
    pyglet_key.DOWN:  'down',
    pyglet_key.SPACE:     'A',   # SPACE = 跳跃
    pyglet_key.LSHIFT:     'B',   # LSHIFT = 冲刺 / 火球
}


def get_action():
    """根据当前按下的键，在 COMPLEX_MOVEMENT 中找到最匹配的动作索引。"""
    buttons = {btn for k, btn in KEY_MAP.items() if k in keys_pressed}
    if not buttons:
        return 0  # NOOP

    # 精确匹配
    for i, action_buttons in enumerate(COMPLEX_MOVEMENT):
        action_set = set(action_buttons) - {'NOOP'}
        if action_set == buttons:
            return i

    # 找最佳子集匹配（不含多余按钮）
    best_action, best_score = 0, -1
    for i, action_buttons in enumerate(COMPLEX_MOVEMENT):
        action_set = set(action_buttons) - {'NOOP'}
        if action_set <= buttons:
            score = len(action_set)
            if score > best_score:
                best_score = score
                best_action = i
    return best_action


# --------------- 参数解析 ---------------
parser = argparse.ArgumentParser(description="手动游玩 / 录制 Mario 动作序列")
parser.add_argument("--record", type=int, default=None,
                    help="录制模式：指定目标 x 坐标，到达后自动保存")
parser.add_argument("--record-all", action="store_true", default=False,
                    help="连续录制所有课程位置的动作序列（一次游玩，依次保存）")
parser.add_argument("--save-dir", type=str, default="./curriculum_actions",
                    help="动作序列保存目录")
cli_args = parser.parse_args()
 
# 确定录制目标列表
if cli_args.record_all:
    # 按 x 升序排列（游玩时依次经过）
    record_targets = [x for x, y in CURRICULUM_POSITIONS]
    recording = True
elif cli_args.record is not None:
    record_targets = [cli_args.record]
    recording = True
else:
    record_targets = []
    recording = False

save_dir = cli_args.save_dir
next_target_idx = 0  # 下一个要保存的目标索引

# --------------- 创建环境 ---------------
import gym_super_mario_bros
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v1', render_mode='human', apply_api_compatibility=True)
env = JoypadSpace(env, COMPLEX_MOVEMENT)

obs = env.reset()

# 先渲染一帧，让 nes_py 创建 pyglet 窗口，然后挂载键盘事件
env.render()

# 穿透所有 wrapper 找到最内层 NES 环境
nes_env = env
while hasattr(nes_env, 'env'):
    nes_env = nes_env.env
viewer = nes_env.viewer
viewer._window.push_handlers(
    on_key_press=_on_key_press,
    on_key_release=_on_key_release,
)

step = 0
recorded_actions = []  # 存储 NES 原始位掩码动作

print("=== 手动游玩 Mario 1-1 ===")
print("方向键: 移动  |  SPACE: 跳跃(A)  |  LSHIFT: 冲刺/火球(B)")
if recording:
    targets_str = ', '.join(str(t) for t in record_targets)
    print(f"\n*** 录制模式: 目标位置 [{targets_str}] ***")
    print("按 R 键重新开始录制")
print("关闭窗口或 Ctrl+C 退出\n")

try:
    while True:
        # R 键重新开始录制
        if recording and pyglet_key.R in keys_pressed:
            keys_pressed.discard(pyglet_key.R)
            print("\n=== 重新开始录制 ===\n")
            recorded_actions.clear()
            next_target_idx = 0
            step = 0
            obs = env.reset()
            continue

        action = get_action()

        # 录制模式下保持无敌状态
        nes_env.ram[0x079E] = 255

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        step += 1

        # 记录 NES 原始位掩码
        nes_bitmask = buttons_to_bitmask(COMPLEX_MOVEMENT[action])
        if recording:
            recorded_actions.append(nes_bitmask)

        # 打印 info 信息
        buttons_str = str(COMPLEX_MOVEMENT[action])
        x_pos = info.get('x_pos', '?')
        remaining = len(record_targets) - next_target_idx if recording else 0
        suffix = f" | 剩余目标: {remaining}" if recording else ""
        print(f"Step {step:>5d} | action: {buttons_str:<25s} | x_pos: {x_pos} | reward: {reward:>6.1f}{suffix}")

        # 录制模式：检查是否到达当前目标
        if recording and not done and next_target_idx < len(record_targets):
            cur_x = info.get('x_pos', 0)
            target_x = record_targets[next_target_idx]
            if cur_x >= target_x - 50:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"target_{target_x}.npy")
                np.save(save_path, np.array(recorded_actions, dtype=np.uint8))
                print(f"\n*** 到达目标! x_pos={cur_x}, 目标={target_x} ***")
                print(f"*** 已保存 {len(recorded_actions)} 步动作到 {save_path} ***")
                next_target_idx += 1
                if next_target_idx >= len(record_targets):
                    print("\n*** 所有目标位置录制完成! ***")
                    break
                else:
                    next_target = record_targets[next_target_idx]
                    print(f"*** 继续前进到下一个目标: x={next_target} ***\n")

        if done:
            if recording:
                print(f"\n--- Episode 结束 (未完成所有目标) ---")
                print("动作已清空，重新开始...\n")
                recorded_actions.clear()
                next_target_idx = 0
                step = 0
            else:
                print(f"\n--- Episode 结束 ---")
                print(f"最终 info: {info}")
                print("重置环境...\n")
            obs = env.reset()

        time.sleep(1 / 60)  # ~60 FPS 节流

except KeyboardInterrupt:
    if recording and recorded_actions:
        # 保存已经过的所有目标（已在循环中保存），提示未完成的
        saved = next_target_idx
        total = len(record_targets)
        print(f"\n手动退出。已保存 {saved}/{total} 个目标位置的动作序列。")
    else:
        print("\n手动退出。")
finally:
    env.close()
