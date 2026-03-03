import gymnasium as gym
import minihack

env = gym.make("MiniHack-Room-5x5-v0",
    observation_keys=("chars_crop", "blstats"))

obs, info = env.reset(seed=42)

# 把地图打印出来看看
print("你看到的世界（9×9 局部视野）：")
for row in obs["chars_crop"]:
    print("  " + "".join(chr(c) if c > 32 else "·" for c in row))

env.close()
