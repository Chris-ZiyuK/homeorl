"""Analyze E3 alpha carryover results."""
import json
import numpy as np

with open('experiments/sequential_results/e3_alpha_carryover/e3_alpha.json') as f:
    data = json.load(f)

agg = data['aggregated']
tasks = data['config']['tasks']
agents = list(agg['phase_end'].keys())

print('=' * 90)
print('E3 ALPHA CARRYOVER — 20 SEED RESULTS')
print('=' * 90)

# 1. Per-task success (diagonal: performance on each task right after training it)
header = f"{'Agent':<22s}"
for t in tasks:
    header += f"{t[:9]:>10s}"
header += f"  {'AVG':>5s}"
print('\n' + header)
print('-' * 115)

agent_avgs = {}
for agent in agents:
    phases = agg['phase_end'][agent]
    line = f"{agent:<22s}"
    scores = []
    for pi, phase in enumerate(phases):
        task = phase['phase_task']
        s = phase['task_metrics'][task]['success_mean']
        scores.append(s)
        line += f"{s:10.2f}"
    avg = np.mean(scores)
    agent_avgs[agent] = avg
    line += f"  {avg:5.2f}"
    print(line)

# 2. Boundary energy
print(f"\nBOUNDARY ENERGY (avg energy at end of each task phase)")
header = f"{'Agent':<22s}"
for t in tasks:
    header += f"{t[:9]:>10s}"
print(header)
print('-' * 115)
for agent in agents:
    phases = agg['phase_end'][agent]
    line = f"{agent:<22s}"
    for phase in phases:
        e = phase.get('policy_boundary_energy_mean', 0) or 0
        line += f"{e:10.1f}"
    print(line)

# 3. Forgetting: reach success after each phase
print(f"\nFORGETTING: Success on 'reach' after each training phase")
header = f"{'Agent':<22s}"
for i in range(len(tasks)):
    header += f"  afterT{i+1:<2d}"
print(header)
print('-' * 115)
for agent in agents:
    phases = agg['phase_end'][agent]
    line = f"{agent:<22s}"
    for phase in phases:
        s = phase['task_metrics']['reach']['success_mean']
        line += f"{s:10.2f}"
    print(line)

# 4. Rankings
print(f"\n{'=' * 60}")
print(f"AGENT RANKINGS (by average task success)")
print(f"{'=' * 60}")
sorted_agents = sorted(agent_avgs.items(), key=lambda x: x[1], reverse=True)
for rank, (agent, avg) in enumerate(sorted_agents, 1):
    bar = '#' * int(avg * 40)
    print(f"  {rank}. {agent:<22s} {avg:.3f}  {bar}")

# 5. Key comparisons
print(f"\n{'=' * 60}")
print(f"KEY COMPARISONS")
print(f"{'=' * 60}")

hace = agent_avgs.get('C_hace', 0)
ewc = agent_avgs.get('F_ewc', 0)
task_only = agent_avgs.get('A_task_only', 0)
hace_ewc = agent_avgs.get('I_hace_ewc', 0)
er = agent_avgs.get('H_er', 0)

print(f"  HACE vs Task-Only:     {hace:.3f} vs {task_only:.3f}  (delta: +{hace-task_only:.3f})")
print(f"  HACE vs EWC:           {hace:.3f} vs {ewc:.3f}  (delta: {hace-ewc:+.3f})")
print(f"  HACE vs ER:            {hace:.3f} vs {er:.3f}  (delta: {hace-er:+.3f})")
print(f"  HACE+EWC vs EWC:       {hace_ewc:.3f} vs {ewc:.3f}  (delta: {hace_ewc-ewc:+.3f})")
print(f"  HACE+EWC vs HACE:      {hace_ewc:.3f} vs {hace:.3f}  (delta: {hace_ewc-hace:+.3f})")

# 6. Per-phase boundary energy for top 3 agents
print(f"\n{'=' * 60}")
print(f"ENERGY TRAJECTORY (top agents)")
print(f"{'=' * 60}")
for agent in ['C_hace', 'I_hace_ewc', 'F_ewc', 'A_task_only']:
    if agent not in agents:
        continue
    phases = agg['phase_end'][agent]
    energies = [phase.get('policy_boundary_energy_mean', 0) or 0 for phase in phases]
    print(f"  {agent:<22s}: {' -> '.join(f'{e:.1f}' for e in energies)}")
