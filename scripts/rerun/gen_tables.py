# Thesis rerun helper (October 2026), copied unchanged from /home/gaga/tamarw1/rerun_scripts/gen_tables.py on rack-shamir.
# Its paths are those of that run (repo checkout, shared data root, /tmp/tw_rerun staging);
# see docs/thesis_results.md.
"""Rows of thesis Tables 3.2-3.4 (LaTeX, same format as the thesis) from a significance.csv. usage: gen_tables.py <csv> <out_json>"""
import sys, json
import pandas as pd
s = pd.read_csv(sys.argv[1])
DS = [("mimic", "MIMIC-III"), ("eicu", "eICU"), ("hirid", "HiRID")]
TK = [("aki", "AKI Stage 3"), ("circulatory_failure", "Circulatory Failure"), ("liver", "Liver ACLF"),
      ("sepsis", "Septic Shock"), ("ventilator", "Ventilator Weaning")]
def p(x): return "$<0.001$" if x < 0.001 else f"{x:.3f}"
def pair(r):
    a, b = f"{r.left_mean:.3f} \pm {r.left_sd:.3f}", f"{r.right_mean:.3f} \pm {r.right_sd:.3f}"
    if round(r.left_mean, 3) > round(r.right_mean, 3): a = f"\mathbf{{{a}}}"
    elif round(r.right_mean, 3) > round(r.left_mean, 3): b = f"\mathbf{{{b}}}"
    return f"${a}$ & ${b}$ & {p(r.p_value)} & {p(r.q_value_bh)}"
def cells(ds, tk, l, r):
    out = []
    for met in ("auroc", "aupr"):
        x = s[(s.dataset == ds) & (s.task == tk) & (s.left == l) & (s.right == r) & (s.metric == met)]
        if len(x) != 1: return None
        out.append(pair(x.iloc[0]))
    return " & ".join(out)
res = {"sd_vs_sdt": {}, "sds_vs_sdst": {}, "mcmc_vs_bootstrap": {}}
for ds, dsl in DS:
    for tk, tkl in TK:
        c = cells(ds, tk, "sd", "sd_traj")
        if c is None: continue
        res["sd_vs_sdt"][f"{ds}/{tk}"] = f"{tkl} & {c} \\\\"
        res["sds_vs_sdst"][f"{ds}/{tk}"] = f"{tkl} & {cells(ds, tk, 'sd_sum', 'sd_sum_traj')} \\\\"
        res["mcmc_vs_bootstrap"][f"{ds}/{tk}/SD+T"] = f"SD+T & {cells(ds, tk, 'sd_traj', 'sd_traj_boot')} \\\\"
        res["mcmc_vs_bootstrap"][f"{ds}/{tk}/SD+S+T"] = f"SD+S+T & {cells(ds, tk, 'sd_sum_traj', 'sd_sum_traj_boot')} \\\\"
json.dump(res, open(sys.argv[2], "w"), indent=1)
print(sum(len(v) for v in res.values()), "rows")
