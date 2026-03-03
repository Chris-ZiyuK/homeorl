import gymnasium as gym
import minihack

# 创建一个简单的 MiniHack 导航任务
env = gym.make("MiniHack-Room-5x5-v0")

obs, info = env.reset()
print("观测空间 keys:", obs.keys())
print("地图形状:", obs["chars"].shape)

# 运行几步随机动作
for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"Step {i+1}: reward={reward}, done={done}")
    if done:
        obs, info = env.reset()
        print("Episode reset!")

env.close()
print("MiniHack 测试成功！")