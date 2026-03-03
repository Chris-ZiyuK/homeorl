import gymnasium as gym
import minihack

env = gym.make("MiniHack-Room-5x5-v0")
obs, info = env.reset()

print(f"动作空间大小: {env.action_space.n} 种动作\n")

ACTIONS = {
    0: "↖ 左上", 1: "↑ 上", 2: "↗ 右上",
    3: "← 左",                5: "→ 右",
    6: "↙ 左下", 7: "↓ 下", 8: "↘ 右下",
    4: "等待(不动)",
}

print("常见动作（移动为主）：")
for idx, name in ACTIONS.items():
    print(f"  动作 {idx} = {name}")

print(f"\n其余动作 (9-{env.action_space.n-1}): 拾取、吃、攻击、施法...")
print("在你的项目中，还会额外加入 QUERY 动作！")

env.close()
