import json

with open('results/evidence/FINAL_EVIDENCE.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

md = []
md.append('# FINAL RESEARCH EVIDENCE (MACHINE-VERIFIED)\n')
md.append(f'**Generated At:** {data["meta"]["generated_at"]}')
md.append(f'**Git Commit:** `{data["meta"]["git_commit"]}`\n')

md.append('## 1. System Environment')
env = data.get('environment', {})
md.append(f'- **OS:** {env.get("os")}')
md.append(f'- **CPU:** {env.get("cpu_model")} ({env.get("logical_cpus")} logical cores)')
md.append(f'- **RAM:** {env.get("ram_gb")} GB')
md.append(f'- **GPU:** {env.get("gpu_name")} ({env.get("gpu_memory_gb")} GB)')
md.append(f'- **PyTorch:** {env.get("pytorch_version")} (CUDA: {env.get("cuda_version")})\n')

md.append('## 2. Model Architecture & Forward Latency (100 passes)')
perf = data.get('performance', {})
arch = perf.get('model_architectures', {})
md.append('| Architecture | Parameters | Forward MS (Mean) | Forward MS (P99) |')
md.append('|---|---|---|---|')
for k, v in arch.items():
    fwd = v.get('forward_pass', v.get('forward_pass_actor', {}))
    pcount = v.get('total_parameters', v.get('total_gnn_parameters', 0))
    md.append(f'| {k} | {pcount:,} | {fwd.get("mean_ms")} | {fwd.get("p99_ms")} |')

md.append('\n## 3. Simulator Performance')
env_bench = perf.get('environment', {})
md.append(f'- **Steps Measured:** {env_bench.get("n_steps_measured")}')
md.append(f'- **Steps per Second:** {env_bench.get("steps_per_second")}')
md.append(f'- **Mean Step Time:** {env_bench.get("mean_step_ms")} ms\n')

md.append('## 4. Forecasting Benchmarks (Test Demand OOD)')
fcast = perf.get('forecasting', {})
md.append('| Model | Fit Time (s) | Predict (mean ms) | MAE | RMSE | sMAPE |')
md.append('|---|---|---|---|---|---|')
for k, v in fcast.items():
    md.append(f'| {k} | {v.get("fit_time_s", "N/A")} | {v.get("predict_mean_ms", "N/A")} | {v.get("MAE")} | {v.get("RMSE")} | {v.get("sMAPE")} |')

md.append('\n## 5. Artifact Validation (Phase 6 / 10)')
p6 = data.get('phase6', {})
md.append(f'- **Phase 6 Checkpoints Discovered:** {p6.get("discovered")}')
md.append(f'- **Phase 6 Checkpoints Valid:** {p6.get("valid")} / {p6.get("expected")}')
md.append(f'- **Hash Mismatches:** {p6.get("hash_mismatches")}')
p10 = data.get('phase10', {})
md.append(f'- **Phase 10 Checkpoints Discovered:** {p10.get("discovered")}')
md.append(f'- **Phase 10 Checkpoints Valid:** {p10.get("valid")} / {p10.get("expected")}\n')

md.append('## 6. Evaluations (Phase 7)')
p7 = data.get('phase7', {})
md.append(f'- **Total Evaluation JSONs:** {p7.get("total_files")}')
md.append(f'- **RL Evaluations Found:** {p7.get("rl_evaluations")} (Expected: {p7.get("expected_rl_evaluations")})')
md.append(f'- **Unique Checkpoints Referenced:** {p7.get("unique_checkpoints_referenced")}')
md.append(f'- **Hash Mismatches:** {p7.get("hash_mismatches")}\n')

md.append('## 7. Statistical Independence (Phase 8)')
p8 = data.get('phase8_statistics', {})
md.append(f'- **Pseudoreplication Check:** {p8.get("pseudoreplication_check")}')
md.append('### Key Comparisons')
for comp in p8.get('statistical_comparisons', []):
    md.append(f'- **{comp["name"]}**')
    md.append(f'  - N1={comp.get("N1")}, N2={comp.get("N2")}')
    md.append(f'  - Welch t: {comp.get("t_statistic", "N/A")}')
    md.append(f'  - Cohen\'s d: {comp.get("cohens_d", "N/A")}')
    md.append(f'  - BH-Adjusted p: {comp.get("bh_adjusted_p", "N/A")}')

md.append('\n## 8. Hard Failures')
fails = data.get('hard_failures', [])
if fails:
    md.append(f'Found {len(fails)} Hard Failures. Below is a sample (up to 20):')
    for f in fails[:20]:
        md.append(f'- X `{f}`')
else:
    md.append('- ✅ No Hard Failures detected in this suite run.')

with open('EVIDENCE.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(md))
