"""
=============================================================================
MiniHack × MDP/POMDP 概念探索
=============================================================================

本脚本通过可运行的代码，逐步展示：
1. MiniHack 环境如何对应 MDP 的五元组 (S, A, T, R, γ)
2. 为什么 MiniHack 实际上是 POMDP（观测 ≠ 状态）
3. 能量扩展 POMDP 如何为你的项目提供基础

每个 Part 都可以单独理解，建议按顺序阅读和运行。
"""

import gymnasium as gym
import minihack
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# PART 1：MDP 的五元组 — 在 MiniHack 中的映射
# ═══════════════════════════════════════════════════════════════════════════
#
# MDP = (S, A, T, R, γ)
#
#  S (状态空间)     = 地图上所有格子 + 玩家属性 + 物品的完整组合
#  A (动作空间)     = 移动、拾取、吃/喝、攻击等
#  T (转移函数)     = env.step(action) — 游戏引擎决定下一个状态
#  R (奖励函数)     = 到达目标得正奖励，每步小惩罚
#  γ (折扣因子)     = 由你在训练时设定（通常 0.99）
#
# 关键理解：env.step() 就是 T(s'|s,a) 的实现！
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1：MDP 五元组在 MiniHack 中的映射")
print("=" * 70)

# 创建一个简单的 5x5 房间环境
env = gym.make(
    "MiniHack-Room-5x5-v0",
    observation_keys=("glyphs", "chars", "blstats", "message",
                      "glyphs_crop", "chars_crop"),
)

# --- S：状态空间 ---
obs, info = env.reset(seed=42)
print("\n【S 状态空间】MiniHack 通过多个 key 描述状态：")
for key, val in obs.items():
    print(f"  {key:15s}  shape={str(val.shape):12s}  dtype={val.dtype}")

# --- A：动作空间 ---
print(f"\n【A 动作空间】共有 {env.action_space.n} 种动作")
# MiniHack 中常见的动作映射（取决于环境配置）
# 0-7: 八方向移动, 其他: 特殊动作（拾取、吃、开门等）
print(f"  动作空间类型: {env.action_space}")
print(f"  示例: 随机动作 = {env.action_space.sample()}")

# --- T：转移函数 = env.step() ---
print("\n【T 转移函数】env.step(action) 就是 T(s'|s,a) 的实现：")
action = 2  # 通常是向下移动
obs_next, reward, terminated, truncated, info = env.step(action)
print(f"  执行动作 {action} 后：")
print(f"  - 获得新观测（新状态的观测）")
print(f"  - reward = {reward}")
print(f"  - terminated = {terminated}  (是否到达终点或死亡)")
print(f"  - truncated = {truncated}   (是否超时)")

# --- R：奖励函数 ---
print(f"\n【R 奖励函数】")
print(f"  每步奖励 = {reward}（通常 -0.01 鼓励快速完成）")
print(f"  到达目标 = +1.0（稀疏奖励！）")
print(f"  这就是你项目中说的'稀疏生存任务'的核心挑战")

env.close()

# ═══════════════════════════════════════════════════════════════════════════
# PART 2：为什么是 POMDP？— 观测 vs 真实状态
# ═══════════════════════════════════════════════════════════════════════════
#
# MDP 假设：智能体看到完整状态 s
# POMDP 现实：智能体只看到观测 o = O(s) ≠ s
#
# 在 MiniHack 中，POMDP 体现在：
# 1. 视野有限：chars 是 21x79 但大部分是空白（未探索区域）
# 2. 裁剪视野：chars_crop 只有 9x9，只看到周围
# 3. 隐藏属性：物品的 blessed/cursed 状态看不到
# 4. blstats 是部分信息：不包含完整游戏状态
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2：MiniHack 是 POMDP — 观测 ≠ 完整状态")
print("=" * 70)

env = gym.make(
    "MiniHack-Room-15x15-v0",  # 更大的房间，更明显的部分可观测性
    observation_keys=("chars", "chars_crop", "blstats", "message"),
)
obs, info = env.reset(seed=42)

# 展示全局地图 vs 局部视野
full_map = obs["chars"]
crop_map = obs["chars_crop"]

print("\n【全局地图 chars (21×79)】— 这是'上帝视角'，包含整个关卡：")
# 只打印有内容的行
for i, row in enumerate(full_map):
    line = "".join(chr(c) if c > 0 else " " for c in row).rstrip()
    if line.strip():
        print(f"  行{i:2d}: {line}")

print(f"\n【局部裁剪 chars_crop (9×9)】— 这是智能体实际'看到'的：")
for row in crop_map:
    line = "".join(chr(c) if c > 0 else " " for c in row)
    print(f"  {line}")

print("\n★ 关键对比：")
print("  全局地图包含整个房间布局（MDP 的'完整状态'）")
print("  裁剪视野只有周围 9×9 区域（POMDP 的'观测'）")
print("  如果你只用 crop 训练，智能体就处于真正的 POMDP 中")
print("  → 同一个 crop 可能对应不同位置（感知混叠/aliasing）")

# blstats 包含什么？
print("\n【blstats (27维)】— 玩家内部状态的部分信息：")
bl = obs["blstats"]
# blstats 的常见字段索引 (NLE 定义):
stat_names = {
    0: "x坐标", 1: "y坐标", 2: "str(力量)", 3: "str_pct",
    4: "dex(敏捷)", 5: "con(体质)", 6: "int(智力)", 7: "wis(智慧)",
    8: "cha(魅力)", 9: "score", 10: "hp(当前生命)", 11: "hp_max(最大生命)",
    12: "depth(层数)", 13: "gold", 14: "energy(魔力)", 15: "energy_max",
    16: "ac(护甲)", 17: "level(等级)", 18: "exp_points",
    19: "exp_level", 20: "time(步数)", 21: "hunger_state(饥饿度!)"
}
for idx, name in stat_names.items():
    if idx < len(bl):
        print(f"  [{idx:2d}] {name:25s} = {bl[idx]}")

print("\n★ 注意 hunger_state（索引21）！")
print(f"  当前饥饿状态 = {bl[21] if len(bl) > 21 else 'N/A'}")
print("  这就是你项目中'能量 E_t'的自然来源！")
print("  但 blstats 不包含：物品详细属性、怪物抗性、未探索区域")
print("  → 所以它是'部分观测'，不是'完整状态'")

env.close()

# ═══════════════════════════════════════════════════════════════════════════
# PART 3：MDP 动力学的代码级理解 — step() 的解剖
# ═══════════════════════════════════════════════════════════════════════════
#
# 每次 env.step(a) 背后发生了什么？
#   1. 游戏引擎接收动作 a
#   2. 计算新状态 s' ~ T(s'|s,a)  ← 这一步我们看不到内部
#   3. 生成观测 o' = O(s')        ← 这就是 obs
#   4. 计算奖励 r = R(s,a)        ← 这就是 reward
#   5. 判断是否终止               ← terminated/truncated
#
# 我们来跑一个完整 episode 观察这个循环：
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3：MDP 动力学 — 一个完整 Episode 的解剖")
print("=" * 70)

env = gym.make(
    "MiniHack-Room-5x5-v0",
    observation_keys=("chars_crop", "blstats", "message"),
    max_episode_steps=50,  # 限制步数
)
obs, info = env.reset(seed=42)

print("\n模拟一个完整 episode（随机策略）：")
print(f"{'步骤':>4s} | {'动作':>4s} | {'奖励':>6s} | {'HP':>4s} | {'位置':>8s} | {'终止':>4s}")
print("-" * 50)

total_reward = 0
for step in range(50):
    action = env.action_space.sample()  # π(a|o) = 随机策略
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    bl = obs["blstats"]
    pos = f"({bl[0]},{bl[1]})"
    hp = bl[10] if len(bl) > 10 else "?"
    done = terminated or truncated

    print(f"{step+1:4d} | {action:4d} | {reward:+6.2f} | {hp:4} | {pos:>8s} | {str(done):>4s}")

    if done:
        result = "死亡" if terminated else "超时"
        print(f"\n  Episode 结束！原因: {result}")
        break

print(f"\n  累积奖励 G = Σr = {total_reward:.3f}")
print(f"  这就是 MDP 中策略 π 的'回报' (return)")
print(f"  RL 的目标：找到 π* 使得 E[G] 最大化")

env.close()

# ═══════════════════════════════════════════════════════════════════════════
# PART 4：从 MDP 到你的项目 — 能量扩展 POMDP 的概念预览
# ═══════════════════════════════════════════════════════════════════════════
#
# 你的项目要做什么？
#
# 标准 MiniHack:
#   状态 s, 动作 a ∈ {移动, 拾取, 吃...}
#   step() → (obs, reward, done)
#
# 你的项目（能量扩展）:
#   扩展状态 s̃ = (s, E_t)        ← 加入能量维度
#   扩展动作 a ∈ {..., QUERY}     ← 加入查询动作
#   E_{t+1} = E_t - c_step + g(s,a) - c_query·1[QUERY]
#   E_t ≤ 0 → death
#
# 这就是一个 Wrapper 的工作！
# 下面用伪代码展示这个概念：
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4：从 MDP 到你的项目 — 能量扩展的概念预览")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────┐
│  标准 MiniHack (MDP/POMDP)                              │
│                                                         │
│  env.step(action)                                       │
│    → obs: {chars, blstats, message}                     │
│    → reward: 到达目标 +1, 每步 -0.01                     │
│    → done: 到达目标 or 死亡 or 超时                      │
│                                                         │
│  动作空间: 移动(8方向) + 拾取 + 吃 + ...                 │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼  加上 SurvivalWrapper
┌─────────────────────────────────────────────────────────┐
│  你的项目（能量扩展 POMDP）                              │
│                                                         │
│  wrapped_env.step(action)                               │
│    → obs: {chars, blstats, message, energy, hint}       │
│    → reward: 同上 + 能量相关                             │
│    → done: 同上 + E_t ≤ 0 (能量耗尽 = death)           │
│                                                         │
│  动作空间: 原始动作 + QUERY（获取 hint，消耗能量）       │
│                                                         │
│  能量动力学:                                             │
│    E_{t+1} = clip(E_t - c_step + eat_gain - c_query)   │
│                                                         │
│  门控机制:                                               │
│    p(QUERY | E_t) = σ(α · (E* - E_t - ε))              │
│    → E_t 低时查询概率↑,  E_t 高时查询概率↓              │
└─────────────────────────────────────────────────────────┘
""")

# 用一个简单的模拟展示能量动力学
print("模拟能量动力学（无真实环境，纯概念演示）：")
print()

E_max = 100
E_star = 70       # 稳态设定点
E = E_max         # 初始能量
c_step = 2        # 每步消耗
c_query = 5       # 查询额外消耗
alpha = 0.15      # 门控斜率
epsilon = 10      # 偏移

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def gate_prob(E_t):
    """内稳态门控：能量越低，查询概率越高"""
    return sigmoid(alpha * (E_star - E_t - epsilon))

print(f"{'步骤':>4s} | {'能量E_t':>7s} | {'偏差D_t':>7s} | {'P(QUERY)':>9s} | {'事件'}")
print("-" * 65)

np.random.seed(42)
for step in range(30):
    D = max(0, E_star - E)        # 稳态偏差
    p_query = gate_prob(E)        # 门控概率

    # 模拟事件
    event = ""
    queried = np.random.random() < p_query
    ate_food = np.random.random() < 0.15  # 15% 概率吃到食物

    E -= c_step  # 每步消耗
    if queried:
        E -= c_query
        event += "🔍QUERY "
    if ate_food:
        E += 20
        event += "🍎+20能量 "
    E = np.clip(E, 0, E_max)

    if not event:
        event = "—"

    bar = "█" * int(E / 2) + "░" * int((E_max - E) / 2)
    print(f"{step+1:4d} | {E:7.1f} | {D:7.1f} | {p_query:9.4f} | {event}")

    if E <= 0:
        print(f"\n  💀 能量耗尽，智能体死亡！")
        break

print(f"""
★ 观察要点：
  1. 能量持续下降（c_step=2/步）→ 生存压力
  2. P(QUERY) 随能量降低而升高 → 内稳态门控
  3. 吃食物恢复能量 → 但需要 QUERY 知道哪个食物安全
  4. QUERY 本身也消耗能量(c_query=5) → 查询有代价

这就是你项目的核心循环。
""")

print("=" * 70)
print("总结：概念映射")
print("=" * 70)
print("""
  MDP 概念          MiniHack 代码              你的项目扩展
  ─────────         ──────────────             ────────────
  S (状态)    →     env 内部状态          →    s̃ = (s, E_t)
  A (动作)    →     env.action_space      →    A ∪ {QUERY}
  T (转移)    →     env.step(a)           →    + 能量动力学
  R (奖励)    →     reward                →    + 生存奖励
  O (观测)    →     obs dict              →    + energy + hint
  
  POMDP 的核心：obs ≠ state
  → chars_crop 只能看到周围 9×9
  → blstats 不包含物品属性/怪物抗性
  → 智能体必须在不完整信息下做决策
  → QUERY 是一种"主动减少不确定性"的动作
""")
