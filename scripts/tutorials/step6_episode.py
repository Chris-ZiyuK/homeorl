import gymnasium as gym
import minihack

env = gym.make("MiniHack-Room-5x5-v0",
    observation_keys=("chars_crop", "blstats"),
    max_episode_steps=30)

total_reward = 0
obs, info = env.reset(seed=42)

print("随机策略的一局游戏：")
print(f"{'步':>3} {'动作':>4} {'奖励':>6} {'累积':>7}")
print("-" * 28)

for step in range(30):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

    print(f"{step+1:3d} {action:4d} {reward:+6.2f} {total_reward:+7.3f}")
    if done:
        reason = "到达目标!" if reward > 0.5 else "超时"
        print(f"\n结果: {reason}")
        break

print(f"\n总奖励 G = {total_reward:.3f}")
print(f"RL的目标：找到策略π*让这个G尽可能大")

env.close()
