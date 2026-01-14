# stats_to_latex.py
import pandas as pd
import numpy as np
from scipy import stats

# load CSV (adjust path if needed)
df = pd.read_csv("experiment_results.csv")

# sanity: ensure 'mode' is a column (avoid df.mode attribute problem)
print("Columns:", df.columns.tolist())

# compute aggregated mean ± std
agg = df.groupby(['map_type','mode']).agg({
    'avg_response_time': ['mean','std'],
    'completed': ['mean','std'],
    'unresponded': ['mean','std'],
    'total_distance': ['mean','std'],
    'utilization': ['mean','std']
})
agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
agg = agg.reset_index()

def fmt(mean, std):
    return f"{mean:.2f} $\\pm$ {std:.2f}"

# build LaTeX table rows
latex_rows = []
for _, r in agg.iterrows():
    row = " & ".join([
        r['map_type'],
        r['mode'],
        fmt(r['avg_response_time_mean'], r['avg_response_time_std']),
        fmt(r['completed_mean'], r['completed_std']),
        fmt(r['unresponded_mean'], r['unresponded_std']),
        fmt(r['total_distance_mean'], r['total_distance_std']),
        f"{r['utilization_mean']:.3f}"
    ]) + r" \\"
    latex_rows.append(row)

latex_table_body = "\n".join(latex_rows)
print("\n=== LaTeX table body (paste into your tabular) ===\n")
print(latex_table_body)

# Paired tests: for each map_type, pair trials by 'trial' and compare GA vs GA+Fuzzy
tests = {}
for mtype in sorted(df['map_type'].unique()):
    sub_ga = df[(df['map_type']==mtype) & (df['mode']=='ga')].sort_values('trial')
    sub_fz = df[(df['map_type']==mtype) & (df['mode']=='ga_fuzzy')].sort_values('trial')
    merged = pd.merge(sub_ga, sub_fz, on='trial', suffixes=('_ga','_fz'))
    if merged.empty:
        print(f"\nNo paired data found for map_type={mtype} (merged empty).")
        continue

    # avg_response_time
    x = merged['avg_response_time_ga']
    y = merged['avg_response_time_fz']
    t_rt, p_rt = stats.ttest_rel(x, y)
    try:
        w_rt, p_w_rt = stats.wilcoxon(x, y)
    except Exception:
        w_rt = p_w_rt = np.nan

    # completed
    x2 = merged['completed_ga']
    y2 = merged['completed_fz']
    t_comp, p_comp = stats.ttest_rel(x2, y2)
    try:
        w_comp, p_w_comp = stats.wilcoxon(x2, y2)
    except Exception:
        w_comp = p_w_comp = np.nan

    tests[mtype] = {
        'avg_rt': {'t': t_rt, 'p': p_rt, 'wilcoxon_p': p_w_rt, 'mean_diff': float((x-y).mean())},
        'completed': {'t': t_comp, 'p': p_comp, 'wilcoxon_p': p_w_comp, 'mean_diff': float((x2-y2).mean())},
        'n_pairs': len(merged)
    }

print("\n=== Paired test summary ===\n")
for mtype, t in tests.items():
    print(f"Map type: {mtype}  (n pairs = {t['n_pairs']})")
    print(f"  Avg RT: mean diff = {t['avg_rt']['mean_diff']:.3f}, paired t = {t['avg_rt']['t']:.3f}, p = {t['avg_rt']['p']:.4f}, wilcoxon p = {t['avg_rt']['wilcoxon_p']}")
    print(f"  Completed: mean diff = {t['completed']['mean_diff']:.3f}, paired t = {t['completed']['t']:.3f}, p = {t['completed']['p']:.4f}, wilcoxon p = {t['completed']['wilcoxon_p']}")
    print()


