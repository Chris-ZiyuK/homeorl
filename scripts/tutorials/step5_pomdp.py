import gymnasium as gym
import minihack

env = gym.make("MiniHack-Room-15x15-v0",
    observation_keys=("chars", "chars_crop", "blstats"))

obs, info = env.reset(seed=42)

# ① 完整地图（上帝视角）
print("① 完整地图 chars (21×79) — 上帝视角：")
for row in obs["chars"]:
    line = "".join(chr(c) if c > 32 else " " for c in row).rstrip()
    if line.strip():
        print(f"  {line}")

# ② 智能体的视野（局部裁剪）
print("\n② 智能体看到的 chars_crop (9×9) — 只有这么多：")
for row in obs["chars_crop"]:
    print("  " + "".join(chr(c) if c > 32 else "·" for c in row))

print("\n★ 关键区别：")
print("  完整地图 = MDP 的'状态 s'（包含一切信息）")
print("  裁剪视野 = POMDP 的'观测 o'（只能看到一部分）")
print("  观测 ≠ 状态 → 这就是为什么是 POMDP！")

env.close()
