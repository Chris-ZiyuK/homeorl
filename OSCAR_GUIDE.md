# HACE × Crafter — Oscar HPC 实验指南

从零开始在 Oscar 上运行 Crafter HACE 实验的完整指南。

---

## 1. 克隆仓库

```bash
ssh zkong10@ssh.ccv.brown.edu
cd ~/codebase   # 或任意你喜欢的目录
git clone git@github.com:Chris-ZiyuK/homeorl.git
cd homeorl
```

---

## 2. 环境设置（一次性）

```bash
bash scripts/setup_oscar_env.sh
```

这个脚本会自动：
- 加载 Python module
- 创建 `.venv` 虚拟环境
- 安装 PyTorch (CUDA) + crafter + stable-baselines3 + 其他依赖
- 运行 Crafter wrapper 和 Gymnasium adapter 的 sanity test

如果看到全部 `✓`，环境就准备好了。

> **注意**：之后每次登录 Oscar 都需要激活虚拟环境：
> ```bash
> cd ~/codebase/homeorl
> source .venv/bin/activate
> ```

---

## 3. Smoke Test（验证环境，~5 分钟）

```bash
source .venv/bin/activate
mkdir -p experiments/crafter/logs
sbatch experiments/crafter/run_crafter_smoke_test.sh
```

这只跑 2 个 job（vanilla + hace），每个 10K 步。完成后检查：

```bash
cat experiments/crafter/results/smoke_*/summary.json
```

---

## 4. 提交 Full Run（1M 步 × 5 agents × 5 seeds）

```bash
sbatch --array=0-8 experiments/crafter/run_crafter_packed.sh
```

### 运行原理

每个 SLURM job 内部**并行启动 3 个实验**（利用 CPU 只有 32% 使用率的事实），总共：

| 参数 | 值 |
|------|---|
| 总实验数 | 25（5 agents × 5 seeds）|
| SLURM jobs | 9（每 job 3 个并行实验）|
| 每 job 资源 | 4 CPU, 6GB RAM, 无 GPU |
| 每 job 时长 | ~3 小时 |
| **总时长** | **~15 小时**（受 2 并发限制，9 jobs 分 5 轮）|

### 实验条件

| Agent | α | β | 测什么 |
|-------|:-:|:-:|--------|
| `vanilla` | 1 | 0 | Baseline：原版 Crafter reward |
| `hace` | 1 | 1 | **我们的方法**：Crafter + 多维 HACE |
| `pure_homeo` | 0 | 1 | 消融：只有 HACE，没有 task reward |
| `health_only` | 1 | 1 | 消融：单维 HACE（health only）|
| `naive_survival` | 1 | — | Sham control：raw vital bonus |

### 输出文件

每个实验的结果保存在 `experiments/crafter/results/full_<agent>_s<seed>/`：

```
full_hace_s0/
├── config.json          # 实验配置
├── episode_data.json    # 每 episode 的详细数据（vitals、death cause 等）
├── summary.json         # 聚合统计
├── model_final.zip      # 训练好的 PPO 模型
├── train.log            # 训练日志
└── tb_logs/             # TensorBoard 日志
```

---

## 5. 监控运行状态

### 方法一：快速查看 SLURM 队列

```bash
squeue -u $USER
# 或
myq
```

### 方法二：查看实验完成情况（推荐 ⭐）

```bash
bash scripts/check_crafter_status.sh
```

输出示例：

```
── Completed Experiments ──
  Agent              s0   s1   s2   s3   s4    Done
  ──────────────────────────────────────────────────
  vanilla            ✓    ✓    ✓    ▶    ·     3/5
  hace               ✓    ✓    ▶    ·    ·     2/5
  pure_homeo         ✓    ▶    ·    ·    ·     1/5
  health_only        ·    ·    ·    ·    ·     0/5
  naive_survival     ·    ·    ·    ·    ·     0/5

  Total: 6/25 completed, 0 failed
  Legend: ✓=done  ▶=running  ✗=failed  ·=not started
```

### 方法三：查看单个实验的实时日志

```bash
# 查看正在运行的实验日志
tail -f experiments/crafter/results/full_hace_s0/train.log

# 查看 SLURM 输出
tail -f experiments/crafter/logs/crafter_hace_*.out
```

### 方法四：查看资源利用率（job 完成后）

```bash
seff <JOB_ID>
```

---

## 6. 分析结果

全部完成后，运行分析脚本：

```bash
python experiments/crafter/analyze_crafter.py \
    --results-dir experiments/crafter/results \
    --latex
```

这会生成：
- 存活时间对比图
- 死因分布图
- Vital levels 对比图
- LaTeX 格式的结果表格

---

## 文件结构

```
homeorl/
├── src/envs/
│   ├── crafter_hace_wrapper.py      # HACE reward wrapper
│   └── crafter_gymnasium_adapter.py # SB3 兼容层
├── experiments/crafter/
│   ├── train_crafter.py             # 训练脚本
│   ├── analyze_crafter.py           # 分析脚本
│   ├── run_crafter_packed.sh        # ⭐ Full run SLURM 脚本
│   ├── run_crafter_smoke_test.sh    # Smoke test
│   └── results/                     # 实验输出
├── scripts/
│   ├── setup_oscar_env.sh           # 环境设置
│   └── check_crafter_status.sh      # ⭐ 状态监控
└── configs/
    └── crafter_experiment.yaml      # 实验配置
```

---

## 常见问题

### Q: Job 提交后马上消失？
检查 SLURM error log：
```bash
cat experiments/crafter/logs/crafter_hace_<JOB_ID>_<TASK>.err
```
常见原因：module load 失败、venv 未激活。

### Q: 某个实验失败了怎么办？
查看对应的 `train.log`：
```bash
cat experiments/crafter/results/full_<agent>_s<seed>/train.log
```
修复后可以单独重跑：
```bash
source .venv/bin/activate
python experiments/crafter/train_crafter.py \
    --agent hace --seed 0 --steps 1000000 \
    --outdir experiments/crafter/results/full_hace_s0 --no-record
```
（注意：需在计算节点上运行，不要在 login node 跑）

### Q: 如何只重跑部分实验？
修改 `sbatch --array=` 参数。例如只跑 job 5（对应 pure_homeo s0, s1, s2）：
```bash
sbatch --array=5 experiments/crafter/run_crafter_packed.sh
```
