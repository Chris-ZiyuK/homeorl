# Homeostatic Grounding: Do Internal Energy States Help RL Agents Learn the Meaning of Symbols?

> **Course:** CS 2951X — Reintegrating Artificial Intelligence  
> **Focus:** Symbol Grounding · Homeostatic RL · Functional Understanding  
> **Date:** February 2026

---

## Abstract

How does an agent learn that fire is dangerous and water is beneficial? In biological organisms, internal homeostatic signals — hunger, pain, energy depletion — provide immediate, salient feedback that naturally grounds abstract concepts in survival-relevant experience. This project investigates whether analogous internal energy states can accelerate **functional symbol grounding** in reinforcement learning agents operating under partial observability.

We design a controlled survival environment containing multiple objects with hidden effects (beneficial, harmful, or neutral). We compare three agent conditions: (A) a baseline agent with only terminal reward, (B) an agent with observable internal energy that provides immediate feedback upon interacting with objects, and (C) an agent with both energy observation and the ability to query an external knowledge source. Our central hypothesis is that **energy-aware agents develop correct object-effect associations significantly faster than terminal-reward-only agents**, and that the addition of queryable knowledge further accelerates this grounding process.

This work connects homeostatic reinforcement learning with symbol grounding theory, proposing that internal states serve as a **natural bridge** from sub-symbolic sensory experience to functional symbolic understanding — a perspective aligned with embodied cognition and the "Reintegrating AI" vision.

---

## 1. Motivation

### 1.1 The Symbol Grounding Problem in RL

Standard RL agents learn entirely from reward signals. In sparse-reward environments, the agent may require thousands of episodes before associating a particular object (e.g., a red potion) with its effect (e.g., HP loss). This is because:

- The reward signal is **delayed**: the agent only learns "that episode was bad" at termination, with no indication of *which* action or object caused the failure.
- The association between **perceptual features** (color, shape, symbol) and **functional effects** (heal, harm, nothing) must be discovered through trial-and-error.
- In POMDP settings, the agent cannot directly observe object properties — it must **infer meaning through interaction**.

### 1.2 Homeostasis as Natural Grounding

In biological organisms, the grounding problem is solved elegantly through **interoception** — the perception of internal body states:

- A child touches a hot stove → **immediate pain signal** → learns "stove = dangerous" in one trial
- An animal eats a toxic berry → **nausea/energy drop** → learns to avoid that berry (Garcia effect)
- Hunger drives foraging; satiation signals "food = good" at the moment of consumption

The key insight is that **internal states transform delayed, sparse environmental feedback into immediate, dense personal experience**. The agent doesn't need to wait until death to learn that poison is bad — the energy drop itself is an immediate grounding signal.

### 1.3 Research Gap

| Existing Work | What It Does | What's Missing |
|---|---|---|
| Homeostatic RL (Keramati & Gutkin 2014) | Models reward as drive reduction | Doesn't study symbol grounding |
| Curiosity-driven (Pathak 2017, Burda 2018) | Intrinsic motivation for exploration | No internal state modeling |
| AFK / Q-BabyAI (Liu et al. 2022) | Agent queries external knowledge | No homeostatic mechanism |
| PAE (Wu et al. 2024) | External knowledge + exploration | No grounding measurement |
| Functional Grounding (Carta et al. 2023) | Defines grounding via environment dynamics | No homeostatic signal study |

**Our unique position:** We combine homeostatic RL with functional grounding measurement. We ask: does adding internal energy states to the agent's observation *cause* faster grounding of object meanings?

---

## 2. Research Questions and Hypotheses

**Central RQ:** Does internal energy observation accelerate functional symbol grounding in survival RL?

### H1: Grounding Acceleration
Energy-aware agents (Group B) learn to correctly distinguish beneficial from harmful objects **significantly faster** (in fewer episodes) than terminal-reward-only agents (Group A).

**Rationale:** Energy change upon object interaction provides immediate, dense feedback that creates a direct association between object symbol and effect.

### H2: Query Amplification
When QUERY access is available, the grounding acceleration is **greater** for energy-aware agents (Group C) than it would be without energy observation, demonstrating an **interaction effect** between internal states and external knowledge.

**Rationale:** Energy observation helps the agent understand *what the hint means* — without internal state feedback, the hint is just an abstract signal.

### H3: Transfer Robustness
After training, when object-effect mappings are changed (e.g., red becomes safe, blue becomes harmful), energy-aware agents **re-adapt faster** because they detect the mismatch through immediate energy feedback.

---

## 3. Experimental Design

### 3.1 Environment: Multi-Object Survival Grid

A 7×7 grid world with:
- **Agent** starting at a random position
- **Exit** at a fixed corner (reaching it = success, +1.0 reward)
- **3-4 unknown objects** scattered in the grid, visually distinguishable (different symbols/colors) but with hidden effects:

| Object | Symbol | Effect | Agent Knows? |
|--------|--------|--------|-------------|
| Type A | `a` (red) | Harmful: E -= 40 | ❌ Must learn |
| Type B | `b` (blue) | Beneficial: E += 30 | ❌ Must learn |
| Type C | `c` (green) | Neutral: E ± 0 | ❌ Must learn |

**Energy dynamics:**
- Initial energy: E₀ = 100
- Per-step cost: c_step = 5
- E ≤ 0 → death (episode terminates, reward = -1.0)
- Agent must eat beneficial objects to survive long enough to reach the exit

**POMDP aspect:** Objects are identified by symbol only. The agent cannot observe their effects until interacting (stepping on them). Each episode randomizes object positions.

### 3.2 Three Experimental Groups

#### Group A: Terminal-Only (Baseline)
- **Observation:** [agent_pos, object_positions, object_types]
- **No** energy in observation
- **Reward:** +1.0 at exit, -1.0 at death, 0 otherwise
- The agent only learns from terminal outcomes. It must die multiple times to infer which objects are harmful.

#### Group B: Energy-Aware (Homeostatic)
- **Observation:** [agent_pos, object_positions, object_types, **energy**]
- Energy is visible as a continuous signal in [0, 1]
- **Reward:** same terminal rewards, **plus** energy change provides implicit shaping
- When the agent eats a harmful object, it immediately sees E drop — this creates a dense, immediate grounding signal.

#### Group C: Energy + QUERY (Full)
- Same as Group B, plus a **QUERY action** (cost: c_query energy)
- QUERY returns a hint about a specific object type (e.g., "type A is harmful")
- Hint is injected into observation as additional dimensions
- Tests whether external knowledge further accelerates grounding when combined with internal states

### 3.3 Metrics

#### Primary: Functional Grounding Score (FGS)
```
FGS(t) = P(approach beneficial) - P(approach harmful)
```
Measured over evaluation episodes at regular intervals. FGS ranges from -1 (approaches harmful, avoids beneficial) to +1 (perfect grounding). This directly measures whether the agent has learned the *meaning* of each object symbol.

#### Secondary:
| Metric | Description |
|--------|------------|
| **Grounding Speed** | Episodes until FGS > 0.5 (consistently correct behavior) |
| **Survival Rate** | Fraction of episodes where agent doesn't die |
| **Exit Rate** | Fraction of episodes where agent reaches exit |
| **Learning Curve AUC** | Area under FGS-vs-episodes curve |

### 3.4 Transfer Test (H3)

After 3000 episodes of training, swap the effects:
- Type A (was harmful) → now beneficial
- Type B (was beneficial) → now harmful

Measure how many episodes each group needs to re-ground (FGS recovers to > 0.5). Energy-aware agents should detect the mismatch faster through immediate energy feedback.

### 3.5 Controls and Ablations

| Ablation | What It Tests |
|----------|--------------|
| Energy visible but no death condition | Is it the energy *signal* or the survival *pressure* that drives grounding? |
| Different c_step (low/medium/high) | Does stronger survival pressure accelerate or harm grounding? |
| More object types (5-6) | Does grounding scale with complexity? |
| Noisy energy (add Gaussian noise) | How robust is energy-based grounding to imperfect interoception? |

---

## 4. Technical Approach

### 4.1 Agent Architecture

All groups share the same DQN architecture (for fairness), differing only in observation dimensions:

```
Input: obs (varies by group)
  → Linear(obs_dim, 128) → ReLU
  → Linear(128, 128) → ReLU
  → Linear(128, n_actions)
Output: Q-values for each action
```

- **Group A:** obs_dim = 8 (positions only)
- **Group B:** obs_dim = 9 (positions + energy)
- **Group C:** obs_dim = 12 (positions + energy + hint)

### 4.2 Training Protocol

- **Algorithm:** DQN with target network, epsilon-greedy exploration
- **Episodes:** 3000 per condition
- **Seeds:** ≥ 10 random seeds per condition
- **Evaluation:** every 100 episodes, run 50 eval episodes (greedy policy) to measure FGS

### 4.3 Measuring Grounding

To compute FGS, during evaluation episodes we track:
1. When agent is adjacent to an object, does it step toward it or away?
2. Count approach/avoid decisions separately for each object type
3. FGS = (approach_beneficial / total_beneficial_encounters) - (approach_harmful / total_harmful_encounters)

---

## 5. Expected Results

```
                    Grounding   Survival   Exit
                    Speed       Rate       Rate
Group A (Terminal)  Slowest     ~40%       ~35%
Group B (Energy)    Faster      ~70%       ~65%  
Group C (Energy+Q)  Fastest     ~85%       ~80%
```

**Key prediction:** The gap between A and B demonstrates that **internal energy states function as a natural grounding mechanism**. The gap between B and C shows that **external knowledge amplifies grounding when combined with internal states**.

---

## 6. Significance and Connection to Reintegrating AI

### 6.1 Why This Matters

This work offers a **computationally minimal** demonstration that homeostatic mechanisms — long studied in cognitive science and neuroscience — have a concrete, measurable benefit for AI:

1. **Symbol grounding through embodiment:** Internal states provide the "body" through which abstract symbols acquire meaning. This connects to embodied cognition theory and Harnad's symbol grounding problem.
2. **Dense signals from sparse environments:** Homeostasis transforms sparse terminal rewards into dense, immediate internal signals — a form of natural reward shaping that doesn't require manual engineering.
3. **Biologically grounded AI design:** Rather than designing reward functions, we let the agent's internal needs create natural meaning — mirroring how biological organisms learn.

### 6.2 Connection to Course Themes

- **Reintegrating symbolic and sub-symbolic:** Internal energy states create a natural bridge — the agent develops symbolic understanding ("red = bad") from sub-symbolic energy dynamics
- **Cognitive science meets ML:** Homeostatic regulation is a well-studied cognitive mechanism; we test its computational utility
- **Against pure end-to-end learning:** We argue that architectural choices (what the agent can observe about itself) matter as much as learning algorithms

---

## 7. Related Work

| Paper | Relevance |
|-------|-----------|
| Keramati & Gutkin 2014 (eLife) | Homeostatic RL formalization — reward as drive reduction |
| Yoshida et al. 2025 | HRRL produces anticipatory regulation and risk aversion |
| Pathak et al. 2017 (ICM) | Curiosity-driven exploration baseline |
| Liu et al. 2022 (AFK, PMLR) | Queryable knowledge source for RL agents |
| Wu et al. 2024 (PAE, ICLR) | External knowledge + exploration efficiency |
| Nam et al. 2021 (ACNO-MDP) | Observation as costly action — formal analogy to QUERY |
| Carta et al. 2023 (GLAM) | Functional grounding = alignment with environment dynamics |
| Harnad 1990 | The symbol grounding problem (foundational) |
| Samvelyan et al. 2021 | MiniHack platform for RL research |

---

## 8. Timeline (12 Weeks)

| Week | Milestone |
|------|-----------|
| 1-2 | Environment v2 implementation (multi-object, 3 groups) |
| 3 | DQN training pipeline + FGS metric implementation |
| 4 | Smoke test: 2-seed run, verify A < B < C trend |
| 5-7 | Full training: 3 groups × 10+ seeds × 3000 episodes |
| 8 | Transfer test (effect swap) |
| 9 | Ablations (c_step sweep, noisy energy, more objects) |
| 10-11 | Paper writing: intro, method, results, analysis |
| 12 | Final revision, code cleanup, reproducibility check |

---

## 9. Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Group B doesn't outperform A | Energy is information — if B ≈ A, it means the agent isn't using energy. Add energy-based reward shaping as a fallback condition. |
| DQN too weak for the task | Switch to PPO or increase network capacity. The task is simple enough that DQN should work. |
| FGS metric noisy | Increase eval episodes (50→200) and seeds (10→20). Use bootstrap CI. |
| "Just reward shaping" criticism | Emphasize that energy is not an engineered reward — it's a natural consequence of the environment's physics. The contribution is the *grounding measurement*, not the reward. |

---

## 10. Deliverables

1. **Paper** (workshop-format, 4-6 pages): problem → formalization → experiment → results → analysis
2. **Code repository**: environment, agents, training scripts, evaluation, plotting
3. **Reproducibility package**: configs, seeds, raw training logs
4. **Main figures** (4):
   - FGS learning curves (3 groups, with 95% CI)
   - Survival rate over training
   - Transfer test: FGS recovery after effect swap
   - Object approach/avoid heatmap per group
