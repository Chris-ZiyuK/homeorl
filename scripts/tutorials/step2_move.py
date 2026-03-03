import gymnasium as gym
import minihack

env = gym.make("MiniHack-Room-5x5-v0",
    observation_keys=("chars_crop", "blstats"))

obs, info = env.reset(seed=42)

def show(obs, label=""):
    print(f"\n{label}")
    for row in obs["chars_crop"]:
        print("  " + "".join(chr(c) if c > 32 else "·" for c in row))
    bl = obs["blstats"]
    print(f"  位置: ({bl[0]}, {bl[1]})")

show(obs, "【移动前】")

# 动作 2 = 向南走
obs, reward, terminated, truncated, info = env.step(2)

show(obs, "【向南走一步后】")
print(f"  奖励: {reward}")

env.close()
