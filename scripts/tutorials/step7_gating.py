import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

E_star = 70   # 能量稳态设定点
alpha = 0.15  # 门控斜率
epsilon = 10  # 偏移量

print("能量 E_t → 查询概率 P(QUERY)")
print("-" * 42)
for E in [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]:
    p = sigmoid(alpha * (E_star - E - epsilon))
    bar = "█" * int(p * 30) + "░" * (30 - int(p * 30))
    print(f"  E={E:3d}  P={p:.3f}  {bar}")

print(f"""
观察：
  能量高(100) → P≈0.00  几乎不查询（没必要）
  能量中(60)  → P≈0.18  开始考虑查询
  能量低(20)  → P≈0.99  几乎一定查询（生存危机！）

这就是内稳态门控：
  能量偏离稳态 → 触发信息获取 → 帮助决策 → 恢复稳态
  就像人饿了会更主动找食物信息一样。""")
