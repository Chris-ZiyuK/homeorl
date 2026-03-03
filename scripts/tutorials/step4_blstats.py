import gymnasium as gym
import minihack

env = gym.make("MiniHack-Room-5x5-v0",
    observation_keys=("chars_crop", "blstats"))

obs, info = env.reset(seed=42)
bl = obs["blstats"]

IMPORTANT = {
    0:  ("x坐标",       bl[0]),
    1:  ("y坐标",       bl[1]),
    10: ("HP (当前)",   bl[10]),
    11: ("HP (最大)",   bl[11]),
    20: ("游戏步数",    bl[20]),
    21: ("饥饿状态",    bl[21]),
}

print("blstats 关键字段（共27维）：\n")
for idx, (name, val) in IMPORTANT.items():
    marker = " ← 你项目的'能量'来源！" if idx == 21 else ""
    print(f"  [{idx:2d}] {name:12s} = {val}{marker}")

print(f"\n饥饿状态含义：")
print(f"  0=正常  1=饥饿  2=虚弱  3=晕倒  4=昏厥  5=饿死")
print(f"\n这就是你项目中 E_t 的自然基础！")

env.close()
